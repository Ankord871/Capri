"""Капри в Telegram.

  python telegram_bot.py
  python telegram_bot.py --token 123:ABC --allow 123456789

ВАЖНО: бот управляет вашим ПК, поэтому слушается только чатов из списка доступа
(--allow или файл telegram_allowed.txt, по одному chat id в строке). Чужим он отвечает
«Доступ закрыт» и пишет их chat id (чтобы вы могли добавить себя). Без списка не выполняется ничего.

Бот принимает голосовые сообщения, распознаёт их той же моделью
и отвечает твоим голосом из папки voices/.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import wave

import numpy as np

try:
    import requests
except ImportError:
    sys.exit("Нужен requests:  pip install requests")

from audio_features import SR, waveform_features
from common import CONFIG_PATH, BASE, build_model, get_classes, load_checkpoint, load_config
from assistant import Assistant
import actions
import tts

VOICES_DIR = os.path.join(BASE, "voices")
API = "https://api.telegram.org/bot{token}"


def convert_to_wav(src_path, dst_path):
    """Любой аудиофайл → 16 kHz mono 16-bit WAV через ffmpeg."""
    import subprocess
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", dst_path],
        capture_output=True, timeout=30,
    )
    if r.returncode != 0 or not os.path.isfile(dst_path):
        raise RuntimeError(f"ffmpeg не смог конвертировать: {r.stderr[-300:]}")


def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(1)
    if sr != SR:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)), np.arange(len(x)), x).astype(np.float32)
    return x


def find_voice_file(key):
    """Ищет voices/<key>.wav, с вариантами для unknown."""
    candidates = [key]
    if key == "unknown":
        candidates = ["unknown", "unknown2"]
    for k in candidates:
        p = os.path.join(VOICES_DIR, f"{k}.wav")
        if os.path.isfile(p):
            return p
    if key in tts.NO_FALLBACK:
        return None
    for fb in ("ok",):
        p = os.path.join(VOICES_DIR, f"{fb}.wav")
        if os.path.isfile(p):
            return p
    return None


class TelegramCapri:
    def __init__(self, token, ckpt="checkpoints/latest.npz", config=CONFIG_PATH, allowed=()):
        self.token = token
        self.allowed = {int(x) for x in allowed}
        self.api = API.format(token=token)
        self.cfg_path = config
        self.load_cfg = lambda: load_config(self.cfg_path)

        cfg = self.load_cfg()
        classes = get_classes(cfg)
        model = build_model(len(classes))
        meta = load_checkpoint(ckpt, model)
        if not meta:
            sys.exit("Нет обученной модели. Сначала: python train.py --fresh")
        if meta.get("classes") != classes:
            sys.exit("Список команд изменился. Перенесите выученное: python train.py --grow")
        print(f"Модель: эпоха {meta['epoch']}, val_acc={meta.get('val_acc')}")

        # Assistant без реального TTS — мы сами шлём голос
        self.asst = Assistant(model, classes, self.load_cfg, self._dummy_say, actions.execute)
        self.offset = 0

    def _dummy_say(self, text, voice=None):
        """Заглушка: Assistant вызывает say, мы перехватываем текст/voice."""
        self._last_reply = (text, voice)

    def _api(self, method, **kwargs):
        r = requests.post(f"{self.api}/{method}", data=kwargs, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(data)
        return data["result"]

    def _download_file(self, file_id, dest):
        info = self._api("getFile", file_id=file_id)
        url = f"https://api.telegram.org/file/bot{self.token}/{info['file_path']}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)

    def _send_voice(self, chat_id, wav_path, caption=None):
        """Отправляет WAV как voice (Telegram сам перекодирует) или как audio."""
        # Сначала пробуем как voice (нужен ogg/opus — конвертируем)
        ogg = wav_path + ".ogg"
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "64k", ogg],
                capture_output=True, timeout=20, check=True,
            )
            with open(ogg, "rb") as f:
                r = requests.post(
                    f"{self.api}/sendVoice",
                    data={"chat_id": chat_id, "caption": caption or ""},
                    files={"voice": ("voice.ogg", f, "audio/ogg")},
                    timeout=30,
                )
            if r.json().get("ok"):
                return
        except Exception as e:
            print(f"[tg] voice ogg не вышел ({e}), шлю как audio")
        # fallback — document/audio
        with open(wav_path, "rb") as f:
            requests.post(
                f"{self.api}/sendAudio",
                data={"chat_id": chat_id, "caption": caption or "", "title": "Капри"},
                files={"audio": ("capri.wav", f, "audio/wav")},
                timeout=30,
            )

    def _send_text(self, chat_id, text):
        self._api("sendMessage", chat_id=chat_id, text=text)

    def handle_voice(self, chat_id, file_id):
        self._last_reply = (None, None)
        self.asst.awake_until = self.asst.clock() + 10      # каждое голосовое — отдельное сообщение: «проснулась» не нужно ждать
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.ogg")
            wav = os.path.join(tmp, "in.wav")
            self._download_file(file_id, src)
            convert_to_wav(src, wav)
            wave_data = read_wav(wav)

            event, name, conf = self.asst.handle(wave_data)
            print(f"  [{event}] {name} {conf:.0%}", flush=True)

            # что ответить
            text, voice_key = self._last_reply
            if not text:
                cfg = self.load_cfg()
                if event == "wake":
                    text, voice_key = cfg["replies"]["wake"], "wake"
                elif event == "command":
                    c = cfg["commands"].get(name, {})
                    text = c.get("reply") or cfg["replies"]["ok"]
                    voice_key = c.get("voice", name)
                elif event == "unknown":
                    text, voice_key = cfg["replies"]["unknown"], "unknown"
                else:
                    return  # silence / ignored

            voice_path = find_voice_file(voice_key) if voice_key else None
            if voice_path:
                self._send_voice(chat_id, voice_path, caption=None)
            else:
                self._send_text(chat_id, text or "…")

    def handle_text(self, chat_id, text):
        text = (text or "").strip().lower()
        if text in ("/start", "старт", "привет"):
            cfg = self.load_cfg()
            phrase = cfg["replies"].get("start", "Привет, я Капри")
            path = find_voice_file("start")
            if path:
                self._send_voice(chat_id, path)
            else:
                self._send_text(chat_id, phrase)
            return
        if text in ("/help", "помощь"):
            self._send_text(
                chat_id,
                "Отправь голосовое сообщение.\n"
                "Сначала скажи «Капри», потом команду:\n"
                "• включи музыку\n• открой браузер\n• громче / тише\n• пауза\n• следующий трек",
            )
            return
        self._send_text(chat_id, "Лучше голосовым 🎤")

    def run(self):
        me = self._api("getMe")
        print(f"Бот @{me.get('username')} запущен. Жду голосовые…", flush=True)
        # приветственный файл уже в voices/start.wav

        while True:
            try:
                updates = self._api(
                    "getUpdates",
                    offset=self.offset,
                    timeout=30,
                    allowed_updates=json.dumps(["message"]),
                )
            except Exception as e:
                print(f"[tg] сеть: {e}, жду 5 с…")
                time.sleep(5)
                continue

            for u in updates:
                self.offset = u["update_id"] + 1
                msg = u.get("message") or {}
                chat_id = msg.get("chat", {}).get("id")
                if not chat_id:
                    continue
                if chat_id not in self.allowed:
                    print(f"[tg] чужой чат {chat_id} — отказ", flush=True)
                    try:
                        self._send_text(chat_id, f"Доступ закрыт. Твой chat id: {chat_id}. "
                                                 f"Владелец может добавить его в telegram_allowed.txt")
                    except Exception:
                        pass
                    continue
                try:
                    if "voice" in msg:
                        self.handle_voice(chat_id, msg["voice"]["file_id"])
                    elif "audio" in msg:
                        self.handle_voice(chat_id, msg["audio"]["file_id"])
                    elif "text" in msg:
                        self.handle_text(chat_id, msg["text"])
                except Exception as e:
                    print(f"[tg] ошибка обработки: {e}")
                    try:
                        self._send_text(chat_id, "Что-то пошло не так, попробуй ещё раз")
                    except Exception:
                        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("CAPRI_TG_TOKEN") or "")
    ap.add_argument("--ckpt", default="checkpoints/latest.npz")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--allow", nargs="*", default=[], help="chat id, которым можно управлять ПК")
    a = ap.parse_args()
    allowed = set(a.allow)
    p = os.path.join(BASE, "telegram_allowed.txt")
    if os.path.isfile(p):
        allowed |= {x.strip() for x in open(p, encoding="utf-8").read().split() if x.strip()}
    try:
        allowed = {int(x) for x in allowed}
    except ValueError:
        sys.exit("chat id — это числа (например 123456789)")
    if not allowed:
        print("Список доступа пуст: бот никого не слушает. Напишите ему из своего Telegram — он ответит вашим chat id, "
              "затем: python telegram_bot.py --allow ВАШ_ID (или впишите id в telegram_allowed.txt)")

    token = a.token.strip()
    if not token:
        # пробуем файл рядом
        for p in ("telegram_token.txt", os.path.join(BASE, "telegram_token.txt")):
            if os.path.isfile(p):
                token = open(p, encoding="utf-8").read().strip()
                break
    if not token:
        sys.exit(
            "Нужен токен бота.\n"
            "  python telegram_bot.py --token 123456:ABC…\n"
            "или положи токен в файл telegram_token.txt"
        )

    # сохраняем токен локально, чтобы не вводить каждый раз
    token_path = os.path.join(BASE, "telegram_token.txt")
    if not os.path.isfile(token_path):
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(token)

    bot = TelegramCapri(token, ckpt=a.ckpt, config=a.config, allowed=allowed)
    bot.run()


if __name__ == "__main__":
    main()
