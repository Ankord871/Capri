"""Общее: пути, конфиг, список классов, сборка модели, надёжные чекпоинты."""
import copy
import hashlib
import json
import os

import numpy as np

from audio_features import FEAT_DIM
from nn import MLP

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")
SPECIAL = ["_silence", "_unknown", "wake"]    # тишина, «чужие слова», имя ассистента (в конфиге: name)
CONFIG_VERSION = 3
HIDDEN = (320, 160)    # признаки теперь с дельтами (в 2 раза крупнее), сеть чуть шире
INIT_SEED = 1234    # ОДИНАКОВЫЙ на всех ПК: сети с разной инициализацией усреднять нельзя

def _cmd(phrase, action, target="", **extra):
    return {"phrase": phrase, "action": action, "target": target, "reply": "Готово", "voice": "ok", **extra}


DEFAULT_CONFIG = {
    "version": CONFIG_VERSION,
    "name": "Капри",
    "browser": "",      # "" = браузер по умолчанию; brave / opera / chrome / edge / firefox или путь к .exe
    "home_url": "https://www.google.com",    # главная страница: её открывает команда «открой браузер»
    "commands": {
        "music":       _cmd("включи музыку",        "open_url"),
        "pause":       _cmd("пауза",                "media_play_pause"),
        "next":        _cmd("следующий трек",       "media_next"),
        "volume_up":   _cmd("громче",               "volume_up"),
        "volume_down": _cmd("тише",                 "volume_down"),
        "mute":        _cmd("без звука",            "volume_mute"),
        "browser":     _cmd("открой браузер",       "open_home"),
        "lock":        _cmd("заблокируй компьютер", "lock_screen"),
    },
    "replies": {
        "ok": "Готово",        # общий ответ на новые команды; его голос — voices/ok.wav
        "start": "Привет, я Капри, твой голосовой помощник. Что будем делать?",
        "wake": "Готова тебя слушать",
        "unknown": "Я тебя не поняла, повтори",
        "no_target": "Цель не задана",
        "error": "Не получилось",
    },
    "audio": {"min_rms": 0.005, "confidence": 0.62, "awake_seconds": 7.0},
    # окно Капри на рабочем столе: corner — top_left / top_right / bottom_left / bottom_right;
    # image — gif из папки avatar/ (по умолчанию capri.gif)
    "hud": {"enabled": True, "corner": "bottom_right", "click_through": False, "image": "capri.gif"},
    # обучение на опыте: услышанное складывается в inbox/, вы исправляете ошибки (терминал: inbox / fix),
    # затем python train.py --more 20. auto_accept: 0 — выкл.; 0.9 — уверенно выполненные команды
    # сразу идут в обучение (быстро, но ошибки закрепляются — включайте осознанно)
    "learning": {"save_inbox": True, "inbox_max": 200, "auto_accept": 0.0},
}


def atomic_write_bytes(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _migrate(user, version):
    """Старый конфиг -> текущий. Ссылки, свои команды, ответы и пороги остаются как были."""
    rep = user.setdefault("replies", {})
    if version < 2:                       # старый конфиг: у каждой команды свой файл голоса (voices/<команда>.wav)
        for name, c in user.get("commands", {}).items():
            c.setdefault("voice", name)
            if name == "browser" and c.get("action") == "open_url" and c.get("target") == "https://www.google.com":
                c["action"], c["target"] = "open_home", ""
    if version < 3 and "ok" not in rep:   # общий ответ новых команд — самый частый из ваших («Готово»)
        used = [c["reply"] for c in user.get("commands", {}).values() if c.get("reply")]
        if used:
            rep["ok"] = max(set(used), key=used.count)


def load_config(path=CONFIG_PATH):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        version = int(user.get("version", 1))
        migrated = version < CONFIG_VERSION
        if migrated:
            _migrate(user, version)
        for k in ("replies", "audio", "hud", "learning"):
            cfg[k].update({a: b for a, b in user.get(k, {}).items() if k != "hud" or a in cfg["hud"]})
        for k in ("name", "browser", "home_url"):
            if k in user:
                cfg[k] = user[k]
        cfg["commands"] = user.get("commands", cfg["commands"])
        if migrated:
            save_config(cfg, path)
    else:
        save_config(cfg, path)
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    atomic_write_bytes(path, json.dumps(cfg, ensure_ascii=False, indent=2).encode("utf-8"))


def get_classes(cfg):
    return SPECIAL + list(cfg["commands"].keys())


def new_command(cfg, phrase, action, target="", **extra):
    """Заготовка новой команды: ответ и голос — общие («ok»), как у остальных новых команд."""
    return {"phrase": phrase, "action": action, "target": target, "reply": cfg["replies"]["ok"], "voice": "ok", **extra}


def classes_hash(classes):
    return hashlib.sha1(json.dumps(classes, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def build_model(n_classes):
    return MLP([FEAT_DIM, *HIDDEN, n_classes], seed=INIT_SEED)


# ---------------- чекпоинты ----------------
def atomic_savez(path, arrays, meta):
    """Пишем во временный файл -> fsync -> подменяем. Предыдущую версию храним как .bak."""
    import io
    arrays = dict(arrays)
    arrays["meta"] = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:       # pickle запрещён: файлы могут прийти по сети
        data = {k: z[k] for k in z.files}
    meta = json.loads(bytes(data.pop("meta")).decode("utf-8"))
    return data, meta


def save_checkpoint(path, model, meta):
    atomic_savez(path, model.state_arrays(), meta)


def load_checkpoint(path, model):
    """Загружает чекпоинт в модель. Если основной файл повреждён — берёт .bak. Возвращает meta или None."""
    for p in (path, path + ".bak"):
        if os.path.exists(p):
            try:
                data, meta = read_npz(p)
                model.load_state(data)
                return meta
            except Exception as e:                      # битый файл (обрыв записи и т.п.)
                print(f"[checkpoint] {p} не читается ({e}); пробую резервную копию")
    return None


def grow_from_checkpoint(path, model, old_classes, new_classes):
    """Переносит выученное в модель с ДРУГИМ списком команд (добавили/убрали/переставили).

    Скрытые слои копируются целиком, выходной слой — по именам классов; у новых классов веса
    почти нулевые, и их предстоит доучить. Возвращает meta или None, если чекпоинт не читается.
    """
    for p in (path, path + ".bak"):
        if not os.path.exists(p):
            continue
        try:
            data, meta = read_npz(p)
        except Exception:
            continue
        L = len(model.W)
        for i in range(L - 1):
            model.W[i], model.b[i] = data[f"W{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32)
            model.mW[i], model.vW[i] = data[f"mW{i}"].copy(), data[f"vW{i}"].copy()
            model.mb[i], model.vb[i] = data[f"mb{i}"].copy(), data[f"vb{i}"].copy()
        old_idx = {n: j for j, n in enumerate(old_classes)}
        rng = np.random.default_rng(INIT_SEED)
        o = L - 1
        for j, name in enumerate(new_classes):
            if name in old_idx:
                k = old_idx[name]
                model.W[o][:, j], model.b[o][j] = data[f"W{o}"][:, k], data[f"b{o}"][k]
                model.mW[o][:, j], model.vW[o][:, j] = data[f"mW{o}"][:, k], data[f"vW{o}"][:, k]
                model.mb[o][j], model.vb[o][j] = data[f"mb{o}"][k], data[f"vb{o}"][k]
            else:
                model.W[o][:, j] = rng.normal(0, 0.01, model.W[o].shape[0]).astype(np.float32)
                model.b[o][j] = 0.0
                model.mW[o][:, j] = model.vW[o][:, j] = 0.0
                model.mb[o][j] = model.vb[o][j] = 0.0
        model.t = int(data["adam_t"])
        return meta
    return None


# ---------------- записи имени ассистента ----------------
def wake_status(data_dir, name):
    """'empty' — записей нет; 'ok' — записаны для этого имени; 'mismatch' — для другого или неизвестно для какого."""
    d = os.path.join(data_dir, "wake")
    if not os.path.isdir(d) or not any(f.lower().endswith(".wav") for f in os.listdir(d)):
        return "empty"
    mp = os.path.join(d, ".name")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            return "ok" if f.read().strip() == name else "mismatch"
    return "mismatch"


def write_wake_marker(data_dir, name):
    d = os.path.join(data_dir, "wake")
    os.makedirs(d, exist_ok=True)
    atomic_write_bytes(os.path.join(d, ".name"), name.encode("utf-8"))
