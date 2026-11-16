"""Логика ассистента: услышал фразу -> классифицировал -> имя («Капри») будит, следующая фраза = команда."""
import time

from actions import Spoken
from audio_features import waveform_features


class Assistant:
    """emit(событие, **данные) — необязательный «слушатель» событий (его использует окно HUD):
    wake, command, unknown, speak, speak_end."""

    def __init__(self, model, classes, load_cfg, say, run_action, clock=time.monotonic, emit=None, on_clip=None):
        """on_clip(wave, событие, класс, уверенность) — вызывается после каждой фразы: так услышанное
        попадает в inbox/ для обучения на опыте (см. learn.py)."""
        self.model, self.classes = model, classes
        self.load_cfg, self._say_fn, self.run_action, self.clock = load_cfg, say, run_action, clock
        self.emit = emit or (lambda *a, **k: None)
        self.on_clip = on_clip
        self.awake_until = 0.0

    def say(self, text, voice=None):
        self.emit("speak", text=text)
        try:
            self._say_fn(text, voice)
        finally:
            self.emit("speak_end")

    def classify(self, wave):
        p = self.model.predict_proba(waveform_features(wave)[None, :])[0]
        i = int(p.argmax())
        return self.classes[i], float(p[i])

    def _wake(self, cfg):
        self.emit("wake", seconds=cfg["audio"]["awake_seconds"])
        self.say(cfg["replies"]["wake"], "wake")
        self.awake_until = self.clock() + cfg["audio"]["awake_seconds"]      # отсчёт — после её ответа

    def handle(self, wave):
        """Обрабатывает одну фразу. Возвращает (событие, класс, уверенность) — удобно для тестов."""
        result = self._handle(wave)
        if self.on_clip:
            try:
                self.on_clip(wave, *result)
            except Exception as e:                       # обучение — дополнение: из-за него работа не должна падать
                print(f"  [inbox: не сохранилось ({e})]", flush=True)
        return result

    def _handle(self, wave):
        name, conf = self.classify(wave)
        cfg = self.load_cfg()
        sure = conf >= cfg["audio"]["confidence"]
        print(f"  [слышу: {name}, {conf:.0%}]", flush=True)
        if name == "_silence":
            return "silence", name, conf
        if self.clock() >= self.awake_until:          # спим: реагируем только на имя
            if name == "wake" and sure:
                self._wake(cfg)
                return "wake", name, conf
            return "ignored", name, conf
        if name == "wake" and sure:                    # уже слушаем, продлеваем
            self._wake(cfg)
            return "wake", name, conf
        if name in cfg["commands"] and sure:
            self.awake_until = 0.0
            ok, status = self.run_action(name, cfg)
            print(f"  [{'выполнено' if ok else 'не вышло'}: {status}]", flush=True)
            c = cfg["commands"][name]
            self.emit("command", name=name, phrase=c.get("phrase", name), ok=ok, status=str(status))
            if ok:
                if isinstance(status, Spoken):          # время и дата: текст каждый раз новый, говорит системный голос
                    self.say(str(status))
                else:
                    self.say(c["reply"], c.get("voice", name))
            else:
                key = "no_target" if status == "no_target" else "error"
                self.say(cfg["replies"][key], key)
            return "command", name, conf
        self.emit("unknown", name=name, conf=conf)
        self.say(cfg["replies"]["unknown"], "unknown")
        return "unknown", name, conf
