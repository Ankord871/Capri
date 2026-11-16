"""Слушатель микрофона. Простой детектор речи по громкости режет поток на отдельные фразы."""
import queue
from collections import deque

import numpy as np

from audio_features import SR

BLOCK = 320          # 20 мс при 16 кГц


class Segmenter:
    """Принимает блоки по 20 мс. Когда фраза закончилась, возвращает её целиком (иначе None)."""

    def __init__(self, min_rms=0.005, pre_roll=12, end_silence=22, max_blocks=140, min_speech=7):
        self.min_rms, self.end_silence, self.max_blocks, self.min_speech = min_rms, end_silence, max_blocks, min_speech
        self.noise = 0.001                      # оценка шума комнаты, подстраивается сама
        self.pre = deque(maxlen=pre_roll)       # 0.2 с «до начала», чтобы не обрезать первый звук
        self.reset()

    def reset(self):
        self.active, self.buf, self.loud, self.quiet, self.speech = False, [], 0, 0, 0
        self.pre.clear()

    def feed(self, block):
        rms = float(np.sqrt(np.mean(block ** 2)))
        thr = max(self.min_rms, self.noise * 3.2)
        if not self.active:
            self.pre.append(block)
            if rms > thr:
                self.loud += 1
                if self.loud >= 2:               # два громких блока подряд = начало речи (отсекает щелчки)
                    self.active, self.buf, self.quiet, self.speech = True, list(self.pre), 0, self.loud
            else:
                self.loud = 0
                self.noise = 0.97 * self.noise + 0.03 * rms
            return None
        self.buf.append(block)
        if rms > thr:
            self.quiet, self.speech = 0, self.speech + 1
        else:
            self.quiet += 1
        if self.quiet >= self.end_silence or len(self.buf) >= self.max_blocks:    # 0.4 с тишины или 2.5 с речи
            utt, ok = np.concatenate(self.buf), self.speech >= self.min_speech
            self.reset()
            return utt if ok else None
        return None


class Listener:
    def __init__(self, handle, load_cfg, device=None, on_level=None):
        """on_level(rms) вызывается на каждый блок звука (50 раз в секунду): им питается индикатор в окне HUD."""
        self.handle, self.load_cfg, self.device, self.on_level = handle, load_cfg, device, on_level

    def run(self, stop):
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            print(f"Нет доступа к микрофону: {e}\n"
                  f"Установите: pip install sounddevice (на Linux ещё: sudo apt install libportaudio2)")
            stop.set()
            return
        q = queue.Queue()
        seg = Segmenter(min_rms=self.load_cfg()["audio"]["min_rms"])
        try:
            with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK,
                                device=self.device, callback=lambda d, f, t, s: q.put(d[:, 0].copy())):
                print(f"Слушаю. Скажите «{self.load_cfg()['name']}», затем команду. Ctrl+C — выход.", flush=True)
                while not stop.is_set():
                    try:
                        block = q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if self.on_level:
                        self.on_level(float(np.sqrt(np.mean(block ** 2))))
                    utt = seg.feed(block)
                    if utt is not None:
                        self.handle(utt)
                        while not q.empty():     # пока Капри говорила, микрофон слышал её саму — выбрасываем
                            q.get_nowait()
                        seg.reset()
        except Exception as e:
            print(f"Ошибка микрофона: {e}")
            stop.set()
