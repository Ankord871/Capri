"""Обучение на опыте: услышанное складывается в inbox/, вы подтверждаете или исправляете,
затем модель дообучается (python train.py --more 20).

Цикл:  inbox -> play 3 -> fix 3 music (или drop 3) -> ... -> python train.py --more 20

Что сохраняется: фразы, на которые Капри проснулась, выполнила команду или не поняла.
Фразы, сказанные, пока она «спит», и тишина не сохраняются.
"""
import os
import time
import wave

import numpy as np

from audio_features import SR
from common import BASE

INBOX = os.path.join(BASE, "inbox")
DATA = os.path.join(BASE, "data")
SAVED_EVENTS = ("wake", "command", "unknown")


def write_wav(path, x):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcm = (np.clip(np.asarray(x, np.float32), -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def save_clip(x, event, name, conf, learning, inbox=INBOX, data_dir=DATA):
    """Сохраняет фразу. Возвращает путь или None. learning — раздел «learning» из конфига."""
    if event not in SAVED_EVENTS:
        return None
    ts = int(time.time() * 1000)
    auto = float(learning.get("auto_accept") or 0)
    if auto > 0 and event in ("wake", "command") and conf >= auto:
        path = os.path.join(data_dir, name, f"auto_{ts}.wav")           # уверенно распознано — сразу в обучение
        write_wav(path, x)
        return path
    if not learning.get("save_inbox", True):
        return None
    path = os.path.join(inbox, f"{ts}__{name}__{int(conf * 100):03d}__{event}.wav")
    write_wav(path, x)
    _prune(inbox, int(learning.get("inbox_max", 200)))
    return path


def _prune(inbox, keep):
    files = sorted(f for f in os.listdir(inbox) if f.endswith(".wav"))
    for f in files[: max(0, len(files) - max(10, keep))]:
        try:
            os.remove(os.path.join(inbox, f))
        except OSError:
            pass


def list_clips(inbox=INBOX):
    """Новые сверху: [{'n', 'path', 'time', 'pred', 'conf', 'event'}]. Файлы с чужими именами пропускаются."""
    if not os.path.isdir(inbox):
        return []
    out = []
    for f in sorted(os.listdir(inbox), reverse=True):
        parts = f[:-4].split("__") if f.endswith(".wav") else []
        if len(parts) != 4 or not parts[0].isdigit() or not parts[2].isdigit():
            continue
        out.append({"n": len(out) + 1, "path": os.path.join(inbox, f), "time": int(parts[0]) / 1000,
                    "pred": parts[1], "conf": int(parts[2]) / 100, "event": parts[3]})
    return out


def get_clip(n, inbox=INBOX):
    clips = list_clips(inbox)
    if not 1 <= n <= len(clips):
        raise ValueError(f"нет записи №{n} (всего {len(clips)}; список: inbox)")
    return clips[n - 1]


def move_clip(clip, cls, data_dir=DATA):
    """Кладёт запись в data/<класс>/ — она участвует в следующем обучении."""
    dest = os.path.join(data_dir, cls, f"fix_{int(clip['time'] * 1000)}.wav")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.replace(clip["path"], dest)
    return dest


def drop_clip(clip):
    os.remove(clip["path"])
