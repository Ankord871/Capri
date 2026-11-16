"""Запись образцов вашего голоса для обучения.

  python record_data.py              # по очереди все классы
  python record_data.py music 40     # 40 записей одного класса

Советы: 30–50 записей на команду; говорите по-разному (громче/тише, ближе/дальше, быстро/медленно).
Класс _unknown — любые ДРУГИЕ слова и фразы (чем разнообразнее, тем меньше ложных срабатываний).
Класс _silence — тишина и обычные шумы комнаты (клавиатура, вентилятор).
"""
import argparse
import os
import shutil
import sys
import time
import wave

import numpy as np

from audio_features import SR
from common import BASE, get_classes, load_config, wake_status, write_wake_marker

SECONDS = 2.5
PROMPTS = {"_silence": "молчите (шум комнаты, дыхание, клавиатура)",
           "_unknown": "любое другое слово или фраза (каждый раз разное)"}


def save_wav(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("count", nargs="?", type=int, default=30)
    ap.add_argument("--data", default=os.path.join(BASE, "data"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        sys.exit(f"Нет доступа к микрофону: {e}\npip install sounddevice (Linux: sudo apt install libportaudio2)")
    cfg = load_config()
    classes = get_classes(cfg)
    todo = [a.name] if a.name else classes
    if a.name and a.name not in classes:
        sys.exit(f"Нет такого класса. Доступны: {classes}")
    device = int(a.device) if a.device and a.device.isdigit() else a.device

    for name in todo:
        d = os.path.join(a.data, name)
        if name == "wake":
            phrase = f"«{cfg['name']}»"
            if wake_status(a.data, cfg["name"]) == "mismatch":
                print(f"Записи слова wake в data/wake сделаны не для имени «{cfg['name']}» (или неизвестно для какого).")
                if input("Убрать их в архив и записать заново? (да/нет) ").strip().lower() in ("да", "y", "yes"):
                    arch = os.path.join(a.data, "_archive", f"wake_{int(time.time())}")
                    os.makedirs(os.path.dirname(arch), exist_ok=True)
                    shutil.move(os.path.join(a.data, "wake"), arch)
                    print(f"  Старые записи перенесены: {arch}")
            write_wake_marker(a.data, cfg["name"])
        else:
            phrase = PROMPTS.get(name) or f"«{cfg['commands'][name]['phrase']}»"
        have = len([f for f in os.listdir(d) if f.endswith(".wav")]) if os.path.isdir(d) else 0
        print(f"\n=== {name}: говорите {phrase}  (записей: {have}, цель: {a.count}) ===")
        while have < a.count:
            s = input(f"[{have + 1}/{a.count}] Enter — записать, q — следующий класс: ").strip().lower()
            if s == "q":
                break
            time.sleep(0.25)
            print("  Говорите…", flush=True)
            x = sd.rec(int(SECONDS * SR), samplerate=SR, channels=1, dtype="int16", device=device)
            sd.wait()
            x = x[:, 0]
            if name != "_silence" and int(np.abs(x).max()) < 400:
                print("  Слишком тихо — повторите (проверьте микрофон).")
                continue
            save_wav(os.path.join(d, f"{int(time.time() * 1000)}.wav"), x)
            have += 1
    print("\nГотово. Теперь: python train.py")


if __name__ == "__main__":
    main()
