"""Речь Капри: сначала ищет готовый голос в voices/, иначе системный TTS.

Положи свои записи:
  voices/start.wav, voices/wake.wav, voices/unknown.wav,
  voices/browser.wav, voices/volume_up.wav, voices/volume_down.wav,
  voices/ok.wav, voices/lock.wav и т.д.
"""
import os
import random
import shutil
import subprocess
import sys
import threading

from common import BASE

VOICES_DIR = os.path.join(BASE, "voices")

# какие ключи могут иметь несколько вариантов
VARIANTS = {
    "unknown": ["unknown", "unknown2"],
}


def _play_wav(path):
    if sys.platform.startswith("win"):
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    elif sys.platform == "darwin":
        subprocess.run(["afplay", path], timeout=30, check=False)
    else:
        for player in ("paplay", "aplay", "ffplay"):
            p = shutil.which(player)
            if p:
                cmd = [p, path]
                if player == "ffplay":
                    cmd = [p, "-nodisp", "-autoexit", "-loglevel", "quiet", path]
                subprocess.run(cmd, timeout=30, check=False)
                return
        raise RuntimeError("нет paplay / aplay / ffplay")


def _windows_say(text):
    ps = r'''
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo }
$ru = $voices | Where-Object { $_.Culture.Name -like 'ru*' } | Select-Object -First 1
if ($ru) { $s.SelectVoice($ru.Name) }
$s.Rate = 0
$s.Speak($env:CAPRI_TEXT)
'''
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        env={**os.environ, "CAPRI_TEXT": text},
        timeout=30,
        check=False,
    )


def _macos_say(text):
    for voice in ("Milena", "Yuri", "Katya", None):
        cmd = ["say"]
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        r = subprocess.run(cmd, timeout=30, check=False, capture_output=True)
        if r.returncode == 0:
            return
    subprocess.run(["say", text], timeout=30, check=False)


def _linux_say(text):
    if shutil.which("RHVoice-test"):
        try:
            subprocess.run(
                ["RHVoice-test", "-p", "Anna", "-o", "play"],
                input=text.encode("utf-8"),
                timeout=30,
                check=False,
            )
            return
        except Exception:
            pass
    for exe, args in (
        ("espeak-ng", ["-v", "ru", "-s", "155", "-a", "130"]),
        ("spd-say", ["-l", "ru"]),
        ("espeak", ["-v", "ru", "-s", "155"]),
    ):
        if shutil.which(exe):
            subprocess.run([exe, *(args or []), text], timeout=30, check=False)
            return
    raise RuntimeError("нет подходящего TTS (установите espeak-ng или RHVoice)")


def _system_say(text):
    if sys.platform.startswith("win"):
        _windows_say(text)
    elif sys.platform == "darwin":
        _macos_say(text)
    else:
        _linux_say(text)


# для этих фраз «Готово» вместо своей записи было бы неверным ответом (например после ошибки)
NO_FALLBACK = {"start", "wake", "unknown", "no_target", "error"}


def _find_voice(voice):
    """Ищет файл голоса. Поддерживает варианты (unknown / unknown2)."""
    if not voice:
        return None
    candidates = list(VARIANTS.get(voice, [voice]))
    # перемешиваем, чтобы не всегда один и тот же
    random.shuffle(candidates)
    for key in candidates:
        path = os.path.join(VOICES_DIR, f"{key}.wav")
        if os.path.isfile(path):
            return path
    if voice in NO_FALLBACK:           # своей записи нет — говорит системный голос, а не чужая фраза
        return None
    # fallback: ok.wav для команд без своей записи
    ok = os.path.join(VOICES_DIR, "ok.wav")
    if os.path.isfile(ok):
        return ok
    return None


def say(text, voice=None, block=True):
    """Произносит фразу.

    block=True (по умолчанию) — ждёт окончания, чтобы микрофон не услышал Капри.
    """
    print(f"Капри: {text}", flush=True)

    def _do():
        try:
            path = _find_voice(voice)
            if path:
                _play_wav(path)
            else:
                _system_say(text)
        except Exception as e:
            print(f"[tts] голос недоступен ({e}); фраза только в тексте", flush=True)

    if block:
        _do()
    else:
        threading.Thread(target=_do, daemon=True).start()
