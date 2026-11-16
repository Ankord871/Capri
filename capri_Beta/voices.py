"""Голос Капри: какие фразы она должна произносить и как превратить ОДНУ запись в готовые файлы.

  python voices.py list                      # какие фразы нужны и каких файлов ещё нет
  python voices.py split запись.wav          # нарезать запись на файлы voices/*.wav

Как записать: прочитайте фразы из `list` ПО ПОРЯДКУ, с паузой около секунды между ними.
Запись должна быть цельным файлом (WAV; mp3/m4a/ogg — если установлен ffmpeg).
Количество найденных фраз должно совпасть с количеством в списке, иначе скрипт ничего не запишет.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

from common import BASE, load_config

VOICES_DIR = os.path.join(BASE, "voices")


def needed(cfg):
    """Список (ключ файла, текст) в порядке, в котором их надо произносить."""
    r = cfg["replies"]
    base = [("start", r["start"]), ("wake", r["wake"]), ("ok", r["ok"]),
            ("unknown", r["unknown"]), ("no_target", r["no_target"]), ("error", r["error"])]
    keys = {k for k, _ in base}
    for name, c in cfg["commands"].items():
        key = c.get("voice", name)
        if key not in keys and c.get("reply") != r["ok"]:
            base.append((key, c["reply"]))
            keys.add(key)
    return base


def report(cfg, voices_dir=VOICES_DIR):
    lines = ["Фразы Капри (читать по порядку):"]
    for i, (key, text) in enumerate(needed(cfg), 1):
        have = os.path.exists(os.path.join(voices_dir, key + ".wav"))
        lines.append(f"  {i}. «{text}»  → voices/{key}.wav  {'есть' if have else 'НЕТ (пока говорит системный голос)'}")
    return "\n".join(lines)


# ---------------- чтение и нарезка ----------------
def _read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError("нужен 16-бит WAV")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(1)
    return x, sr


def read_audio(path):
    try:
        return _read_wav(path)
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("Не удалось прочитать файл как 16-бит WAV. Сохраните его как WAV или установите ffmpeg (для mp3/m4a/ogg).")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "x.wav")
        r = subprocess.run([ffmpeg, "-y", "-i", path, "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", out],
                           capture_output=True)
        if r.returncode != 0:
            sys.exit("ffmpeg не смог прочитать файл")
        return _read_wav(out)


def find_phrases(x, sr, gap=0.7, min_len=0.15, pad=0.12):
    """Режет запись на фразы по паузам. Возвращает [(начало, конец)] в отсчётах."""
    hop = int(sr * 0.01)
    n = len(x) // hop
    if n == 0:
        return []
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1))
    noise = float(np.percentile(rms, 10))
    thr = max(noise * 4.0, float(rms.max()) * 0.04)
    active = rms > thr
    runs, i = [], 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in runs:                                   # склеиваем куски, между которыми пауза короче gap (запятая внутри фразы)
        if merged and (r[0] - merged[-1][1]) * 0.01 < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    merged = [r for r in merged if (r[1] - r[0]) * 0.01 >= min_len]
    out = []
    for k, (a, b) in enumerate(merged):
        lo = merged[k - 1][1] if k else 0
        hi = merged[k + 1][0] if k + 1 < len(merged) else n
        a2 = max(a - int(pad * 100), (lo + a) // 2)
        b2 = min(b + int(pad * 100), (b + hi) // 2)
        out.append((a2 * hop, b2 * hop))
    return out


def save_wav(path, x, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def split(path, cfg, gap=0.7, out_dir=VOICES_DIR, dry=False):
    """Возвращает список записанных файлов или None, если число фраз не совпало."""
    x, sr = read_audio(path)
    segs = find_phrases(x, sr, gap=gap)
    need = needed(cfg)
    print(f"Найдено фраз в записи: {len(segs)}, нужно: {len(need)}")
    if len(segs) != len(need):
        for i, (a, b) in enumerate(segs, 1):
            print(f"  найдено {i}: {a / sr:.1f}–{b / sr:.1f} с")
        print("Не совпало, ничего не записано. Если фраз найдено БОЛЬШЕ — внутри фразы слишком длинная пауза: "
              "попробуйте --gap 1.0. Если МЕНЬШЕ — паузы между фразами короткие: --gap 0.4 или перезапишите с паузой в секунду.")
        return None
    written = []
    for (a, b), (key, text) in zip(segs, need):
        print(f"  «{text}»  {(b - a) / sr:.1f} с → voices/{key}.wav")
        if not dry:
            p = os.path.join(out_dir, key + ".wav")
            save_wav(p, x[a:b], sr)
            written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("list", help="какие фразы нужны")
    s = sub.add_parser("split", help="нарезать одну запись на файлы")
    s.add_argument("file")
    s.add_argument("--gap", type=float, default=0.7, help="пауза между фразами, с (по умолчанию 0.7)")
    s.add_argument("--dry", action="store_true", help="только показать, ничего не записывать")
    a = ap.parse_args()
    cfg = load_config()
    if a.mode == "list":
        print(report(cfg))
    else:
        sys.exit(0 if split(a.file, cfg, a.gap, dry=a.dry) is not None else 1)


if __name__ == "__main__":
    main()
