"""Самопроверка всего конвейера на СИНТЕТИЧЕСКОМ звуке (тоновые «слова»), без микрофона.

  python selftest.py

Проверяет: обучение, автовозобновление после «выключения питания», защиту от битого чекпоинта,
синхронизацию двух ПК через сервер, отказ на плохие запросы, нарезку речи, логику Капри и терминал.
Это тест механики, а не качества распознавания настоящей речи.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import actions                                   # noqa: E402
import hud                                       # noqa: E402
import learn                                     # noqa: E402
import federated                                 # noqa: E402
import terminal                                  # noqa: E402
import voices                                    # noqa: E402
from assistant import Assistant                  # noqa: E402
from audio_features import SR, waveform_features  # noqa: E402
from common import (DEFAULT_CONFIG, build_model, grow_from_checkpoint, classes_hash, get_classes, load_checkpoint,  # noqa: E402
                    load_config, read_npz, save_config, wake_status, write_wake_marker)
from listener import BLOCK, Segmenter            # noqa: E402
from record_data import save_wav                 # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""), flush=True)


# ---------------- синтетические «слова» ----------------
def class_spec(k):
    rs = np.random.default_rng(100 + k)
    return [(rs.uniform(250, 900), rs.uniform(1000, 3200)) for _ in range(3)]


def synth(rng, classes, name, scale=1.0, dur=2.0):
    k = classes.index(name)
    x = rng.normal(0, rng.uniform(0.001, 0.004), int(dur * SR))
    if name == "_silence":
        return x
    spec = [(rng.uniform(250, 900), rng.uniform(1000, 3200)) for _ in range(3)] if name == "_unknown" else class_spec(k)
    cursor, amp = rng.uniform(0.3, 0.8), rng.uniform(0.1, 0.3)
    for f1, f2 in spec:
        m = int(rng.uniform(0.14, 0.22) * SR)
        t = np.arange(m) / SR
        s = (np.sin(2 * np.pi * f1 * scale * rng.uniform(0.95, 1.05) * t)
             + 0.6 * np.sin(2 * np.pi * f2 * scale * rng.uniform(0.95, 1.05) * t)) * np.hanning(m)
        i0 = int(cursor * SR)
        x[i0:i0 + m] += amp * s[: max(0, len(x) - i0)]
        cursor += m / SR + rng.uniform(0.01, 0.05)
    return x


def make_dataset(root, classes, n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    for name in classes:
        for i in range(n):
            x = np.clip(synth(rng, classes, name, scale), -1, 1)
            save_wav(os.path.join(root, name, f"{i}.wav"), (x * 32767).astype(np.int16))
    write_wake_marker(root, "Капри")


def run_train(args, cwd=HERE):
    return subprocess.run([sys.executable, "train.py", *args], cwd=cwd, capture_output=True, text=True, timeout=600)


def ckpt_meta(path):
    return read_npz(path)[1]


def accuracy_of(weights, classes, scale, n=12, seed=999):
    m = build_model(len(classes))
    m.load_weights(weights)
    rng = np.random.default_rng(seed)
    ok = tot = 0
    for ci, name in enumerate(classes):
        for _ in range(n):
            p = m.predict_proba(waveform_features(synth(rng, classes, name, scale))[None, :])[0]
            ok += int(p.argmax() == ci)
            tot += 1
    return ok / tot


def main():
    tmp = tempfile.mkdtemp(prefix="capri_selftest_")
    cfg_path = os.path.join(tmp, "config.json")
    save_config(DEFAULT_CONFIG, cfg_path)
    cfg = load_config(cfg_path)
    classes = get_classes(cfg)
    dA, dB = os.path.join(tmp, "dataA"), os.path.join(tmp, "dataB")
    t0 = time.time()
    make_dataset(dA, classes, 30, seed=1, scale=1.0)
    make_dataset(dB, classes, 30, seed=2, scale=1.08)       # «второй ПК»: другой микрофон/голос
    print(f"Синтетические данные: {len(classes)} классов × 30 ({time.time() - t0:.0f} с)")

    # ---- 1. обучение + «выключили питание» + продолжение ----
    print("\n1. Чекпоинты и возобновление")
    ck = os.path.join(tmp, "ck1")
    common = ["--data", dA, "--ckpt", ck, "--config", cfg_path, "--epochs", "40"]
    r = run_train(common + ["--stop-after", "4"])
    check("обрыв на эпохе 4 (как выключение ПК)", r.returncode == 3, f"код {r.returncode}")
    m4 = ckpt_meta(os.path.join(ck, "latest.npz"))
    check("чекпоинт сохранён: эпоха 4", m4["epoch"] == 4)
    r = run_train(common)
    check("продолжение с эпохи 4", "Продолжаю с чекпоинта: эпоха 4" in r.stdout, r.stderr[-200:])
    m = ckpt_meta(os.path.join(ck, "latest.npz"))
    check("дообучилось до конца", m["epoch"] == 40, f"эпоха {m['epoch']}")
    check("точность на проверке ≥ 85%", (m["val_acc"] or 0) >= 0.85, f"{m['val_acc']:.1%}")

    probe = build_model(len(classes))
    open(os.path.join(ck, "latest.npz"), "wb").write(b"broken")          # битый основной файл
    got = load_checkpoint(os.path.join(ck, "latest.npz"), probe)
    check("битый чекпоинт -> берётся .bak", got is not None and got["epoch"] == 39, f"эпоха {got and got['epoch']}")
    shutil.copyfile(os.path.join(ck, "latest.npz.bak"), os.path.join(ck, "latest.npz"))

    # ---- 2. синхронизация двух ПК ----
    print("\n2. Сервер и два ПК")
    srv = federated.make_server(os.path.join(tmp, "server_state"), classes, "t0k", "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    ch = classes_hash(classes)
    good = federated.pack_weights(build_model(len(classes)).weights_dict())

    def post(body, token="t0k", chash=ch, name="evil"):
        req = urllib.request.Request(url + "/update", data=body, method="POST", headers={
            "X-Token": token, "X-Classes": chash, "X-Worker": name, "X-Samples": "5"})
        try:
            return urllib.request.urlopen(req, timeout=10).status
        except urllib.error.HTTPError as e:
            return e.code

    bad_shape = federated.pack_weights({k: v[:1] for k, v in build_model(len(classes)).weights_dict().items()})
    nan_w = build_model(len(classes)).weights_dict()
    nan_w["W0"][0, 0] = np.nan
    check("неверный токен -> 403", post(good, token="x") == 403)
    check("чужая форма весов -> 400", post(bad_shape) == 400)
    check("NaN в весах -> 400", post(federated.pack_weights(nan_w)) == 400)
    check("мусор вместо npz -> 400", post(b"not a zip") == 400)
    check("другой список команд -> 409", post(good, chash="zzz") == 409)
    check("плохое имя ПК -> 400", post(good, name="../x") == 400)
    check("состояние сервера не испорчено", srv.agg.round == 0)

    base = ["--config", cfg_path, "--epochs", "40", "--server", url, "--token", "t0k", "--sync-every", "2"]
    procs = []
    for nm, d in (("pcA", dA), ("pcB", dB)):
        procs.append(subprocess.Popen([sys.executable, "train.py", "--data", d, "--ckpt", os.path.join(tmp, "ck_" + nm),
                                       "--worker", nm, *base], cwd=HERE, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True))
    outs = [p.communicate(timeout=600)[0] for p in procs]
    check("оба ПК отработали", all(p.returncode == 0 for p in procs), "; ".join(o[-150:] for o in outs if "Готово" not in o))
    st = srv.agg.status()
    check("сервер получил обновления от обоих", set(st["workers"]) == {"pcA", "pcB"} and st["round"] >= 20, str(st))
    gw = srv.agg.global_w
    accA, accB = accuracy_of(gw, classes, 1.0), accuracy_of(gw, classes, 1.08)
    check("общая модель узнаёт «голоса» обоих ПК (≥ 85%)", min(accA, accB) >= 0.85, f"A {accA:.1%}, B {accB:.1%}")
    r2 = run_train(["--data", dA, "--ckpt", os.path.join(tmp, "ck_new"), "--config", cfg_path, "--epochs", "1",
                    "--server", url, "--token", "t0k", "--worker", "pcC"])
    check("новый ПК стартует с общей модели", "Стартую с общей модели" in r2.stdout, r2.stdout[-200:])
    check("после перезапуска сервер помнит ПК", len(federated.Aggregator(os.path.join(tmp, "server_state"), classes).slots) >= 2)
    srv.shutdown()

    # ---- 3. нарезка речи ----
    print("\n3. Слушатель")
    rng = np.random.default_rng(5)
    noise = lambda s: rng.normal(0, 0.002, int(s * SR))
    burst = lambda s: noise(s) + 0.2 * np.sin(2 * np.pi * 800 * np.arange(int(s * SR)) / SR)
    stream = np.concatenate([noise(0.6), burst(0.6), noise(0.9), burst(0.5), noise(0.9)]).astype(np.float32)
    seg, utts = Segmenter(), []
    for i in range(0, len(stream) - BLOCK, BLOCK):
        u = seg.feed(stream[i:i + BLOCK])
        if u is not None:
            utts.append(len(u) / SR)
    check("две фразы найдены отдельно", len(utts) == 2 and all(0.5 < d < 1.4 for d in utts), str([round(d, 2) for d in utts]))
    seg2 = Segmenter()
    click = np.zeros(len(noise(1.0)), np.float32)
    click[8000] = 0.9
    got = [seg2.feed((noise(1.0) + click).astype(np.float32)[i:i + BLOCK]) for i in range(0, SR - BLOCK, BLOCK)]
    check("одиночный щелчок не считается речью", all(g is None for g in got))

    # ---- 4. логика Капри ----
    print("\n4. Капри: «Капри» -> команда")
    model = build_model(len(classes))
    load_checkpoint(os.path.join(ck, "latest.npz"), model)
    said, opened = [], []

    class FakeWeb:
        @staticmethod
        def open(url):
            opened.append(url)
            return True

    actions.webbrowser = FakeWeb
    for k in list(actions.ACTIONS):                   # тест не должен трогать вашу систему (звук, блокировку экрана)
        if k not in ("open_url", "open_home", "sequence"):
            actions.ACTIONS[k] = lambda cmd, cfg: "заглушка"
    clock = {"t": 100.0}
    a = Assistant(model, classes, lambda: cfg, lambda text, voice=None: said.append(text), actions.execute, lambda: clock["t"])
    rng = np.random.default_rng(77)
    import contextlib
    import io

    def trials(name, awake, n=10):
        """n попыток со свежими образцами; модель не идеальна, поэтому проверяем долю, а не один образец."""
        evs = []
        for _ in range(n):
            a.awake_until = clock["t"] + 10 if awake else 0.0
            with contextlib.redirect_stdout(io.StringIO()):
                evs.append(a.handle(synth(rng, classes, name)))
        return evs

    ev = trials("browser", awake=False)
    check("команда без «Капри» игнорируется (≥ 9 из 10)", sum(e[0] == "ignored" for e in ev) >= 9 and not opened)
    ev = trials("wake", awake=False)
    check("«Капри» будит (≥ 9 из 10), ответ «Готова тебя слушать»", sum(e[0] == "wake" for e in ev) >= 9 and "Готова тебя слушать" in said)
    said.clear()
    opened.clear()
    ev = trials("browser", awake=True)
    done = sum(e[:2] == ("command", "browser") for e in ev)
    check("проснулся -> команда выполняется (≥ 9 из 10)", done >= 9 and len(opened) == done
          and set(opened) == {"https://www.google.com"}, f"{done}/10")
    check("после команды отвечает «Готово»", "Готово" in said)
    said.clear()
    ev = trials("music", awake=True)
    check("цель не задана -> «Цель не задана» (≥ 9 из 10)", sum(e[:2] == ("command", "music") for e in ev) >= 9
          and said.count("Цель не задана") >= 9)
    said.clear()
    ev = trials("_unknown", awake=True)
    rej = sum(e[0] == "unknown" for e in ev)
    check("чужие фразы чаще всего отвергаются («Я тебя не поняла, повтори»)", rej >= 7 and "Я тебя не поняла, повтори" in said, f"{rej}/10")
    a.awake_until = clock["t"] + 5
    clock["t"] += 30
    with contextlib.redirect_stdout(io.StringIO()):
        ev = [a.handle(synth(rng, classes, "browser"))[0] for _ in range(5)]
    check("через 30 с «Капри» забыта", ev.count("ignored") >= 4)
    with contextlib.redirect_stdout(io.StringIO()):
        ev = [a.handle(synth(rng, classes, "_silence"))[0] for _ in range(10)]
    check("тишина не вызывает реакции", ev.count("silence") >= 9)

    # ---- 5. терминал ----
    print("\n5. Мини-терминал")
    tts_said = []
    terminal.tts.say = lambda text, voice=None: tts_said.append(text)
    sh = terminal.CapriShell(cfg_path, inbox=os.path.join(tmp, 'inbox'), data_dir=os.path.join(tmp, 'dataLearn'))
    cmd_cfg = lambda: load_config(cfg_path)["commands"]
    sh.onecmd("set music https://example.com/playlist")
    check("set: ссылка сохранена", cmd_cfg()["music"]["target"] == "https://example.com/playlist")
    sh.onecmd("set music javascript:alert(1)")
    sh.onecmd("set music file:///etc/passwd")
    check("set: не-http ссылки отклонены", cmd_cfg()["music"]["target"] == "https://example.com/playlist")
    sh.onecmd("set volume_up https://example.com")
    check("set: действию без цели ссылку не дать", cmd_cfg()["volume_up"]["target"] == "")
    sh.onecmd("add rock open_url https://example.com/rock")
    check("add: своя команда со своей ссылкой", cmd_cfg().get("rock", {}).get("target") == "https://example.com/rock")
    n = len(cmd_cfg())
    sh.onecmd("add rock open_url https://x.com")
    sh.onecmd("add wake open_url https://x.com")
    sh.onecmd("add ../bad open_url https://x.com")
    check("add: дубликаты и служебные имена отклонены", len(cmd_cfg()) == n)
    opened.clear()
    sh.onecmd("run rock")
    check("run: ссылка открылась, Капри ответила", opened == ["https://example.com/rock"] and tts_said[-1] == "Готово")
    sh.onecmd("run music")
    check("run: открыта ссылка на музыку", opened[-1] == "https://example.com/playlist")
    sh.onecmd("list")


    # ---- 6. браузер, главная страница, готовые команды, сценарии ----
    print("\n6. Браузер, главная страница, готовые команды, сценарии")
    launched = []
    actions._launch = lambda browser, url: launched.append((os.path.basename(browser), url))
    fake_dir = os.path.join(tmp, "fakebrowsers")
    os.makedirs(fake_dir)
    brave, opera = os.path.join(fake_dir, "brave-fake"), os.path.join(fake_dir, "opera-fake")
    open(brave, "w").close()
    open(opera, "w").close()
    sh.onecmd(f"browser {brave}")
    check("browser: общий браузер выбран", load_config(cfg_path)["browser"] == brave)
    opened.clear()
    launched.clear()
    sh.onecmd("run music")
    check("музыка открывается в выбранном браузере, не в системном",
          launched == [("brave-fake", "https://example.com/playlist")] and not opened, str(launched))
    sh.onecmd("home https://example.com/start")
    launched.clear()
    sh.onecmd("run browser")
    check("«открой браузер» открывает главную страницу в нём", launched == [("brave-fake", "https://example.com/start")], str(launched))
    sh.onecmd(f"browser {opera} music")
    launched.clear()
    sh.onecmd("run music")
    sh.onecmd("run browser")
    check("у музыки свой браузер, у главной общий",
          launched == [("opera-fake", "https://example.com/playlist"), ("brave-fake", "https://example.com/start")], str(launched))
    sh.onecmd("browser /нет/такого/браузера")
    sh.onecmd("home javascript:alert(1)")
    check("несуществующий браузер и чужая ссылка отклонены",
          load_config(cfg_path)["browser"] == brave and load_config(cfg_path)["home_url"] == "https://example.com/start")
    sh.onecmd("browser volume_up")
    sh.onecmd(f"browser {opera} volume_up")
    check("браузер нельзя назначить команде без ссылок", "browser" not in cmd_cfg()["volume_up"])
    sh.onecmd("browser default music")
    sh.onecmd("browser default")
    opened.clear()
    launched.clear()
    sh.onecmd("run browser")
    check("default возвращает системный браузер", opened == ["https://example.com/start"] and not launched, f"{opened} {launched}")

    before = len(cmd_cfg())
    sh.onecmd("preset youtube")
    sh.onecmd("preset screenshot")
    sh.onecmd("preset nope")
    sh.onecmd("preset youtube")
    cc = cmd_cfg()
    check("preset: готовые команды добавлены", cc["youtube"]["target"] == "https://www.youtube.com"
          and cc["screenshot"]["action"] == "screenshot" and cc["screenshot"]["phrase"] == "сделай скриншот")
    check("preset: повтор и неизвестный отклонены", len(cc) == before + 2)

    real_sleep = actions.time.sleep
    actions.time.sleep = lambda x: None
    sh.onecmd('scene work browser music "фраза=рабочий режим"')
    w = cmd_cfg()["work"]
    check("scene: сценарий создан", w["action"] == "sequence" and w["steps"] == ["browser", "music"] and w["phrase"] == "рабочий режим")
    opened.clear()
    sh.onecmd("run work")
    check("сценарий: оба шага выполнены по порядку", opened == ["https://example.com/start", "https://example.com/playlist"], str(opened))
    n = len(cmd_cfg())
    sh.onecmd("scene bad browser нет_такой")
    sh.onecmd("scene work2 work music")
    sh.onecmd("scene volume_up browser music")
    check("scene: неизвестный шаг, вложенный сценарий, занятое имя отклонены", len(cmd_cfg()) == n)
    sh.onecmd("remove music")
    check("remove: команду из сценария удалить нельзя", "music" in cmd_cfg())
    sh.onecmd("add empty_link open_url")
    sh.onecmd("scene s2 browser empty_link")
    check("шаг без цели -> «Цель не задана»", actions.execute("s2", load_config(cfg_path)) == (False, "no_target"))
    actions.time.sleep = real_sleep
    sh.onecmd("actions")

    # ---- 7. перенос старого конфига ----
    print("\n7. Перенос конфига версии 1")
    import json
    old_path = os.path.join(tmp, "old_config.json")
    json.dump({"commands": {
        "music": {"phrase": "включи музыку", "action": "open_url", "target": "https://open.spotify.com/x", "reply": "Готово"},
        "browser": {"phrase": "открой браузер", "action": "open_url", "target": "https://www.google.com", "reply": "Готово"},
        "lock": {"phrase": "заблокируй", "action": "lock_screen", "target": "", "reply": "Готово"},
        "mine": {"phrase": "моя", "action": "open_url", "target": "https://a.b", "reply": "Сделано"}},
        "replies": {"start": "Привет, я Капри", "wake": "Готова тебя слушать", "unknown": "Не поняла",
                    "no_target": "Цель не задана", "error": "Не получилось"},
        "audio": {"min_rms": 0.006, "confidence": 0.6, "awake_seconds": 6.0}}, open(old_path, "w", encoding="utf-8"), ensure_ascii=False)
    oc = load_config(old_path)
    check("версия и имя обновлены", oc["version"] == 3 and oc["name"] == "Капри" and oc["replies"]["start"] == "Привет, я Капри" and oc["replies"]["ok"] == "Готово")
    check("ссылка на музыку и свой порог сохранены", oc["commands"]["music"]["target"] == "https://open.spotify.com/x"
          and oc["audio"]["confidence"] == 0.6)
    check("«открой браузер» стала главной страницей", oc["commands"]["browser"]["action"] == "open_home")
    check("свой ответ не потерян, свой голос отдельный", oc["commands"]["mine"]["reply"] == "Сделано"
          and oc["commands"]["mine"]["voice"] == "mine" and oc["commands"]["lock"]["voice"] == "lock")
    check("классы после переноса те же (чекпоинт подходит)", get_classes(oc)[:3] == ["_silence", "_unknown", "wake"]
          and list(oc["commands"]) == ["music", "browser", "lock", "mine"])

    # ---- 8. записи имени и голос ----
    print("\n8. Имя в записях и голос Капри")
    dC = os.path.join(tmp, "dataC")
    save_wav(os.path.join(dC, "wake", "a.wav"), np.zeros(4000, np.int16))
    check("записи без метки считаются чужими", wake_status(dC, "Капри") == "mismatch")
    write_wake_marker(dC, "Другое")
    check("записи другого имени не подходят для «Капри»", wake_status(dC, "Капри") == "mismatch")
    r = run_train(["--data", dC, "--ckpt", os.path.join(tmp, "ck_c"), "--config", cfg_path, "--epochs", "1"])
    check("train отказывается смешивать имена", r.returncode != 0 and "не для имени" in (r.stderr + r.stdout), (r.stderr or r.stdout)[-160:])
    write_wake_marker(dC, "Капри")
    check("после метки «Капри» записи принимаются", wake_status(dC, "Капри") == "ok")
    check("пустая папка — не ошибка", wake_status(os.path.join(tmp, "нет"), "Капри") == "empty")

    base_cfg = load_config(os.path.join(tmp, "fresh_config.json"))
    need = voices.needed(base_cfg)
    check("нужно 6 фраз для голоса", [k for k, _ in need] == ["start", "wake", "ok", "unknown", "no_target", "error"]
          and need[0][1].startswith("Привет, я Капри"), str(need))
    rng2 = np.random.default_rng(3)
    SRV = 24000

    def tone(sec, f):
        t = np.arange(int(sec * SRV)) / SRV
        return 0.3 * np.sin(2 * np.pi * f * t) * np.minimum(1, np.minimum(t, sec - t) * 40)

    gapz = lambda sec: rng2.normal(0, 0.002, int(sec * SRV))
    parts = [gapz(0.8)]
    for i in range(6):
        parts.append(tone(0.6, 300 + 60 * i))
        if i == 2:                                    # пауза внутри фразы, как запятая в «Готово»
            parts += [gapz(0.3), tone(0.4, 500)]
        parts.append(gapz(1.2))
    rec = np.concatenate(parts).astype(np.float32)
    rec_path = os.path.join(tmp, "voice_rec.wav")
    save_wav_sr = lambda p, x, sr: voices.save_wav(p, x, sr)
    save_wav_sr(rec_path, rec, SRV)
    vdir = os.path.join(tmp, "voices_out")
    got = voices.split(rec_path, base_cfg, out_dir=vdir)
    ok_files = got is not None and sorted(os.listdir(vdir)) == sorted(k + ".wav" for k, _ in need)
    check("split: одна запись разрезана на 6 файлов с правильными именами", ok_files, str(got))
    if ok_files:
        import wave as wv
        with wv.open(os.path.join(vdir, "ok.wav")) as wf:
            dur = wf.getnframes() / wf.getframerate()
        check("пауза-запятая внутри фразы не режет её (длина сохраняется)", 1.2 < dur < 1.9, f"{dur:.2f} с")
    bad_dir = os.path.join(tmp, "voices_bad")
    cut = np.concatenate(parts[:-4]).astype(np.float32)
    save_wav_sr(os.path.join(tmp, "short.wav"), cut, SRV)
    check("split: если фраз не хватает — ничего не записывается", voices.split(os.path.join(tmp, "short.wav"), base_cfg, out_dir=bad_dir) is None
          and not os.path.exists(bad_dir))
    terminal.tts.say = lambda text, voice=None: None
    import contextlib as cl
    buf = io.StringIO()
    with cl.redirect_stdout(buf):
        sh.onecmd("voices")
    check("voices: показывает фразы и каких файлов нет", "Готова тебя слушать" in buf.getvalue() and "НЕТ" in buf.getvalue())

    # ---- 9. новые команды, сочетания клавиш, обучение на опыте ----
    print("\n9. Новые команды и обучение на опыте")
    sh.onecmd("preset desktop calc downloads time date switch newtab nope")
    cc = cmd_cfg()
    check("preset: несколько за раз, неизвестный пропущен", all(k in cc for k in ("desktop", "calc", "downloads", "time", "date", "switch", "newtab"))
          and "nope" not in cc)
    check("новые команды используют общий голос ok", cc["calc"]["voice"] == "ok" and cc["calc"]["reply"] == "Готово")
    st = actions._tell_time(None, None)
    check("время называется текстом (Spoken)", isinstance(st, actions.Spoken) and len(st) > 3, repr(st))
    sh.onecmd("add hk hotkey ctrl+shift+t")
    sh.onecmd("add hk2 hotkey ctrl+nonsense")
    check("hotkey: верное сочетание принято, неверное отклонено", cmd_cfg().get("hk", {}).get("target") == "ctrl+shift+t" and cmd_cfg().get("hk2", {}).get("target") == "")
    check("сочетание разбирается", actions.parse_hotkey("ctrl+shift+t") is not None)
    cfg9 = load_config(cfg_path)
    classes9 = get_classes(cfg9)

    inbox = os.path.join(tmp, "inbox")
    dl = os.path.join(tmp, "dataLearn")
    x = synth(np.random.default_rng(1), classes, "wake")
    p1 = learn.save_clip(x, "wake", "wake", 0.91, cfg9["learning"], inbox, dl)
    p2 = learn.save_clip(x, "unknown", "calc", 0.40, cfg9["learning"], inbox, dl)
    p3 = learn.save_clip(x, "ignored", "calc", 0.90, cfg9["learning"], inbox, dl)
    check("inbox: сохраняются пробуждение/команда/непонятое, не «игнор»", p1 and p2 and p3 is None and len(learn.list_clips(inbox)) == 2)
    p4 = learn.save_clip(x, "command", "music", 0.95, {"auto_accept": 0.9}, inbox, dl)
    check("auto_accept: уверенные сразу в обучение", p4 and os.path.dirname(p4) == os.path.join(dl, "music"))
    buf = io.StringIO()
    with cl.redirect_stdout(buf):
        sh.onecmd("inbox")
    check("inbox показывает записи", "поняла как" in buf.getvalue())
    sh.onecmd("fix 1 pause")           # самая новая запись -> pause
    check("fix: запись стала примером команды", len(os.listdir(os.path.join(dl, "pause"))) == 1 and len(learn.list_clips(inbox)) == 1)
    sh.onecmd("fix 1 нет_такого")
    check("fix: неизвестный класс отклонён", len(learn.list_clips(inbox)) == 1)
    sh.onecmd("drop 1")
    check("drop: запись удалена", len(learn.list_clips(inbox)) == 0)
    sh.onecmd("drop 5")

    # grow: добавили команды — выученное переносится
    m1 = build_model(len(classes))
    load_checkpoint(os.path.join(ck, "latest.npz"), m1)
    grown = list(classes) + ["extra_cmd"]
    m2 = build_model(len(grown))
    meta2 = grow_from_checkpoint(os.path.join(ck, "latest.npz"), m2, classes, grown)
    X = np.stack([synth(np.random.default_rng(50 + i), classes, c) for i, c in enumerate(classes[:6]) for _ in range(3)]) if False else None
    rr = np.random.default_rng(31)
    ws = [synth(rr, classes, "wake") for _ in range(12)] + [synth(rr, classes, "browser") for _ in range(12)]
    F = np.stack([waveform_features(w) for w in ws])
    same = (m1.predict_proba(F).argmax(1) == np.array([classes.index(c) for c in ["wake"] * 12 + ["browser"] * 12])).mean()
    after = (m2.predict_proba(F).argmax(1) == np.array([grown.index(c) for c in ["wake"] * 12 + ["browser"] * 12])).mean()
    check("grow: выученное сохранилось после добавления команды", meta2 is not None and after >= same - 0.1 and after >= 0.8, f"{same:.0%} -> {after:.0%}")
    ck2 = os.path.join(tmp, "ck_grow")
    make_dataset(os.path.join(tmp, "dataG"), grown, 30, seed=3)
    cfgG = os.path.join(tmp, "cfgG.json")
    gc = load_config(cfg_path)
    gc["commands"] = {k: v for k, v in list(gc["commands"].items())[:8]}
    gc["commands"]["extra_cmd"] = {"phrase": "x", "action": "volume_mute", "target": "", "reply": "Готово", "voice": "ok"}
    # исходный чекпоинт ck обучен для classes (DEFAULT + добавленное в тестах может отличаться): берём его классы из meta
    mm = ckpt_meta(os.path.join(ck, "latest.npz"))
    gc["commands"] = {k: v for k, v in DEFAULT_CONFIG["commands"].items()}
    gc["commands"]["extra_cmd"] = {"phrase": "x", "action": "volume_mute", "target": "", "reply": "Готово", "voice": "ok"}
    save_config(gc, cfgG)
    shutil.copytree(ck, ck2)
    r = run_train(["--data", os.path.join(tmp, "dataG"), "--ckpt", ck2, "--config", cfgG])
    check("train без --grow отказывается и подсказывает", r.returncode != 0 and "--grow" in (r.stdout + r.stderr))
    r = run_train(["--data", os.path.join(tmp, "dataG"), "--ckpt", ck2, "--config", cfgG, "--grow", "--more", "6"])
    mg = ckpt_meta(os.path.join(ck2, "latest.npz"))
    check("train --grow: доучил с переносом", r.returncode == 0 and "Перенесено выученное" in r.stdout and "extra_cmd" in mg["classes"], (r.stdout + r.stderr)[-250:])
    r = run_train(["--data", os.path.join(tmp, "dataG"), "--ckpt", ck2, "--config", cfgG, "--more", "2"])
    mg2 = ckpt_meta(os.path.join(ck2, "latest.npz"))
    check("train --more: продолжает с чекпоинта", r.returncode == 0 and mg2["epoch"] == mg["epoch"] + 2, f"{mg['epoch']} -> {mg2['epoch']}")

    # ---- 10. окно Капри (модель и gif без экрана) ----
    print("\n10. Окно и персонаж")
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        Image = None
    clk = {"t": 100.0}
    hm = hud.HudModel(lambda: {"name": "Капри", "hud": {}}, clock=lambda: clk["t"])
    check("окно: в покое полупрозрачна, обычная скорость", not hm.active(clk["t"]) and hm.alpha(clk["t"]) < 1 and hm.speed(clk["t"]) == 1.0)
    hm.post("wake", seconds=7.0)
    hm.pump()
    check("окно: «Капри» -> оживлена и непрозрачна", hm.active(clk["t"]) and hm.alpha(clk["t"]) == 1.0 and hm.speed(clk["t"]) > 1)
    clk["t"] += 20
    check("окно: через время снова покой", not hm.active(clk["t"]))
    gif = Image.open(os.path.join(HERE, "avatar", "capri.gif")) if Image else None
    fr = [f.convert("RGBA") for f in ImageSequence.Iterator(gif)] if gif else None
    if fr:
      check("gif персонажа: много кадров, есть прозрачность", gif.n_frames >= 24 and fr[0].getpixel((0, 0))[3] == 0
          and any(f.getpixel((f.width // 2, f.height // 2))[3] > 0 for f in fr))
      check("gif персонажа: кадры различаются (есть движение)", fr[0].tobytes() != fr[gif.n_frames // 2].tobytes())
    dh = load_config(os.path.join(tmp, "nonexistent_cfg.json"))["hud"]
    check("конфиг: настройки окна и learning по умолчанию есть", dh["image"] == "capri.gif" and "learning" in load_config(cfg_path))
    old = os.path.join(tmp, "oldhud.json")
    save_config({**DEFAULT_CONFIG, "hud": {"enabled": False, "mode": "auto", "avatar": "builtin", "title": "X"}}, old)
    check("старые ключи hud (mode/avatar/title) отброшены, enabled сохранён", load_config(old)["hud"] == {**dh, "enabled": False})

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nИтог: {sum(RESULTS)}/{len(RESULTS)} проверок пройдено")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
