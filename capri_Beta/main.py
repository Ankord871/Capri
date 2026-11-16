"""Запуск Капри:  python main.py              (слушать + окно на рабочем столе)
                 python main.py --no-hud     (без окна, только консоль)
                 python main.py --terminal   (+ мини-терминал в этой же консоли)"""
import argparse
import os
import sys
import threading

import actions
import learn
import tts
from assistant import Assistant
from common import BASE, CONFIG_PATH, build_model, get_classes, load_checkpoint, load_config, wake_status
from listener import Listener


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/latest.npz")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--device", default=None, help="номер или имя микрофона (по умолчанию — системный)")
    ap.add_argument("--terminal", action="store_true")
    ap.add_argument("--no-hud", action="store_true", help="не показывать окно на рабочем столе")
    a = ap.parse_args()

    cfg = load_config(a.config)
    classes = get_classes(cfg)
    model = build_model(len(classes))
    meta = load_checkpoint(a.ckpt, model)
    if not meta:
        sys.exit("Нет обученной модели. Сначала: python record_data.py, затем python train.py")
    if meta["classes"] != classes:
        sys.exit("Список команд изменился после обучения. Перенесите выученное: python train.py --grow "
                 "(или с нуля: python train.py --fresh)")
    if wake_status(os.path.join(BASE, "data"), cfg["name"]) == "mismatch":
        print(f"ВНИМАНИЕ: записи слова «wake» сделаны не для имени «{cfg['name']}». "
              f"Запишите заново: python record_data.py wake, затем python train.py --fresh")
    print(f"Модель загружена: эпоха {meta['epoch']}, точность на проверке {meta.get('val_acc')}")

    device = int(a.device) if a.device and a.device.isdigit() else a.device
    load = lambda: load_config(a.config)

    hud_model = window = None
    if not a.no_hud and cfg.get("hud", {}).get("enabled", True):
        try:
            import hud
            hud_model = hud.HudModel(load)
            window = hud.HudWindow(hud_model, load, on_quit=lambda: stop.set())
        except Exception as e:                     # нет tkinter или дисплея — работаем без окна
            print(f"[hud] окно недоступно ({e}); работаю без него")
            hud_model = window = None
    emit = (lambda ev, **k: hud_model.post(ev, **k)) if hud_model else None

    def on_clip(wave, event, name, conf):
        learn.save_clip(wave, event, name, conf, load().get("learning", {}))

    asst = Assistant(model, classes, load, tts.say, actions.execute, emit=emit, on_clip=on_clip)
    listener = Listener(asst.handle, load, device, on_level=hud_model.set_level if hud_model else None)
    stop = threading.Event()

    def start_phrase():
        if emit:
            emit("start")
        tts.say(cfg["replies"]["start"], "start")

    threading.Thread(target=start_phrase, daemon=True).start()
    worker = threading.Thread(target=listener.run, args=(stop,), daemon=True)
    worker.start()
    if a.terminal:
        from terminal import CapriShell
        threading.Thread(target=lambda: (CapriShell(a.config).cmdloop(), stop.set()), daemon=True).start()
    try:
        if window:
            window.run()                           # окно — в главном потоке (требование tkinter)
        else:
            while not stop.is_set():
                stop.wait(0.3)
    except KeyboardInterrupt:
        pass
    stop.set()


if __name__ == "__main__":
    main()
