"""Обучение с чекпоинтами и (по желанию) синхронизацией между ПК.

  python train.py                       # учиться на data/, чекпоинты в checkpoints/
  python train.py --server http://IP:8765 --token ТОКЕН   # + слать веса на главный ПК

Выключили ПК / закрыли окно: просто запустите ту же команду, обучение продолжится
с последнего сохранённого эпохи (веса, состояние Adam, номер эпохи).
"""
import argparse
import math
import os
import sys
import time
import wave

import numpy as np

from audio_features import SR, waveform_features
from common import (CONFIG_PATH, build_model, classes_hash, get_classes, grow_from_checkpoint,
                    load_checkpoint, load_config, read_npz, save_checkpoint, wake_status)
from federated import FederatedClient


def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2 or n == 0:
        raise ValueError("нужен 16-бит PCM WAV, не пустой")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(1)
    if sr != SR:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)), np.arange(len(x)), x).astype(np.float32)
    return x


def load_dataset(data_dir, classes):
    waves, labels = [], []
    for ci, name in enumerate(classes):
        d = os.path.join(data_dir, name)
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if fn.lower().endswith(".wav"):
                try:
                    waves.append(read_wav(os.path.join(d, fn)))
                    labels.append(ci)
                except Exception as e:
                    print(f"[data] пропускаю {name}/{fn}: {e}")
    return waves, np.array(labels, dtype=np.int64)


def augment(w, rng, noise_bank):
    """Случайные искажения: скорость, громкость, шум комнаты, сдвиг. Нужны, чтобы не зазубрить записи."""
    f = rng.uniform(0.9, 1.1)
    n = max(400, int(len(w) / f))
    w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    w = w * rng.uniform(0.4, 1.8)
    sig = float(np.sqrt(np.mean(w ** 2))) + 1e-8
    amp = sig * 10 ** (-rng.uniform(8, 30) / 20)
    if noise_bank and rng.random() < 0.7:
        nz = noise_bank[rng.integers(len(noise_bank))]
        nz = np.resize(nz, len(w))
        nz = nz / (np.sqrt(np.mean(nz ** 2)) + 1e-8)
    else:
        nz = rng.normal(0, 1, len(w))
    w = w + amp * nz
    return waveform_features(w, jitter=int(rng.integers(-1600, 1601)))


def split(labels, n_classes):
    rng = np.random.default_rng(0)
    tr, va = [], []
    for c in range(n_classes):
        idx = rng.permutation(np.nonzero(labels == c)[0])
        k = max(1, len(idx) // 5) if len(idx) >= 5 else 0
        va += list(idx[:k])
        tr += list(idx[k:])
    return np.array(tr), np.array(va)


def accuracy(model, X, y):
    return float((model.predict_proba(X).argmax(1) == y).mean()) if len(y) else float("nan")


def sync(client, model, n_samples, classes, pull):
    ch = classes_hash(classes)
    r = client.push(model.weights_dict(), n_samples, ch)
    if r is None:
        print("[sync] сервер недоступен — продолжаю локально, отправлю в следующий раз")
        return None
    msg = f"[sync] отправлено на главный ПК (раунд {r['round']}, ПК в сети: {r['workers']})"
    if pull:
        g = client.pull(model.weights_dict(), ch)
        if g:
            model.load_weights(g[0])
            msg += "; получена общая модель"
    print(msg)
    return r["round"]


def peek_meta(path):
    """meta чекпоинта (или .bak) без загрузки в модель; None, если нет/битый."""
    for p in (path, path + ".bak"):
        if os.path.exists(p):
            try:
                return read_npz(p)[1]
            except Exception:
                continue
    return None


def train(a):
    cfg = load_config(a.config)
    classes = get_classes(cfg)
    if wake_status(a.data, cfg["name"]) == "mismatch":
        sys.exit(f"Записи слова wake сделаны не для имени «{cfg['name']}» (или неизвестно для какого). "
                 f"Если они от «{cfg['name']}»: создайте файл data/wake/.name с этим именем.\n"
                 f"Иначе перезапишите: python record_data.py wake (предложит убрать старые в архив).")
    waves, labels = load_dataset(a.data, classes)
    counts = {c: int((labels == i).sum()) for i, c in enumerate(classes)}
    missing = [c for c, n in counts.items() if n < 5]
    if missing:
        sys.exit(f"Мало записей (нужно хотя бы 5, лучше 30+) для: {missing}\n"
                 f"Запишите: python record_data.py   (сейчас: {counts})")
    tr, va = split(labels, len(classes))
    noise_bank = [waves[i] for i in tr if labels[i] == 0]
    Xv = np.stack([waveform_features(waves[i]) for i in va]) if len(va) else np.zeros((0, 1))
    yv = labels[va]
    print(f"Классов: {len(classes)}, записей: {len(labels)} (учим на {len(tr)}, проверяем на {len(va)})")

    model = build_model(len(classes))
    ckpt = os.path.join(a.ckpt, "latest.npz")
    client = FederatedClient(a.server, a.token, a.worker) if a.server else None
    start, meta = 0, None
    if not a.fresh:
        old = peek_meta(ckpt)
        if old and old["classes"] != classes:
            if not a.grow:
                sys.exit("Список команд изменился после последнего чекпоинта.\n"
                         "  python train.py --grow    — перенести выученное и доучить новые команды (рекомендуется)\n"
                         "  python train.py --fresh   — начать заново")
            meta = grow_from_checkpoint(ckpt, model, old["classes"], classes)
            if meta:
                start = 0
                a.epochs = a.more or 25
                print(f"Перенесено выученное. Новых/изменённых команд: "
                      f"{[c for c in classes if c not in old['classes']]}. Доучиваю {a.epochs} эпох.")
        else:
            meta = load_checkpoint(ckpt, model)
            if meta:
                start = int(meta["epoch"])
                print(f"Продолжаю с чекпоинта: эпоха {start}")
                if a.more:
                    a.epochs = start + a.more
    if meta is None and client and start == 0:
        g = client.pull(model.weights_dict(), classes_hash(classes))
        if g:
            model.load_weights(g[0])
            print(f"Стартую с общей модели главного ПК (раунд {g[1]})")
    meta = {"classes": classes, "epoch": start, "val_acc": None}
    if start >= a.epochs:
        print(f"Уже обучено до эпохи {start}. Дообучить ещё: python train.py --more 20")
        return meta

    for epoch in range(start, a.epochs):
        t0 = time.time()
        erng = np.random.default_rng(a.seed * 100003 + epoch)
        order = tr[erng.permutation(len(tr))]
        X = np.stack([augment(waves[i], erng, noise_bank) for i in order])
        y = labels[order]
        lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * epoch / max(1, a.epochs - 1))))
        losses = []
        for s in range(0, len(order), a.batch):
            loss, gW, gb = model.loss_and_grads(X[s:s + a.batch], y[s:s + a.batch], erng)
            model.step(gW, gb, lr)
            losses.append(loss)
        val = accuracy(model, Xv, yv)
        meta.update(epoch=epoch + 1, val_acc=val, loss=float(np.mean(losses)))
        print(f"эпоха {epoch + 1}/{a.epochs}  loss {np.mean(losses):.3f}  проверка {val:.1%}  "
              f"({time.time() - t0:.1f} с)", flush=True)
        save_checkpoint(ckpt, model, meta)
        if client and (epoch + 1) % a.sync_every == 0:
            rnd = sync(client, model, len(tr), classes, not a.no_pull)
            if rnd is not None:
                meta["fed_round"] = rnd
                save_checkpoint(ckpt, model, meta)
        if a.stop_after and epoch + 1 >= a.stop_after:
            os._exit(3)                       # проверка: имитируем внезапное выключение питания
    if client:
        sync(client, model, len(tr), classes, not a.no_pull)
        save_checkpoint(ckpt, model, meta)
    print(f"Готово. Чекпоинт: {ckpt}")
    return meta


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--ckpt", default="checkpoints")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--more", type=int, default=0, help="доучить ещё N эпох поверх чекпоинта (после inbox/fix)")
    p.add_argument("--grow", action="store_true", help="список команд изменился: перенести выученное, доучить новые")
    p.add_argument("--fresh", action="store_true", help="игнорировать чекпоинт, начать заново")
    p.add_argument("--server", default=None, help="адрес главного ПК, например http://192.168.1.10:8765")
    p.add_argument("--token", default="")
    p.add_argument("--worker", default=None, help="имя этого ПК (латиница)")
    p.add_argument("--sync-every", type=int, default=2, help="отправлять веса каждые N эпох")
    p.add_argument("--no-pull", action="store_true", help="не забирать общую модель обратно")
    p.add_argument("--stop-after", type=int, default=0, help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
