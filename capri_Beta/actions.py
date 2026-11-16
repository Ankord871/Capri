"""Что Капри умеет делать на ПК.

Безопасность: нет shell=True и нет произвольных команд. Запускается только то, что вы сами
задали в терминале (ссылка http(s), путь к программе или папке), и только по распознанной голосовой команде.
"""
import datetime
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from urllib.parse import urlparse

ACTIONS = {}
NEEDS_TARGET = {"open_url": "ссылка http(s)", "open_app": "путь к программе", "open_folder": "путь к папке",
                "hotkey": "сочетание клавиш, например ctrl+t"}


class Spoken(str):
    """Статус действия, который Капри должна произнести вслух (время, дата)."""
# open_home берёт ссылку из home_url (команда `home` в терминале); sequence — из списка шагов (команда `scene`)


class NeedTarget(Exception):
    """У команды не задана цель (например, ссылка на музыку)."""


def action(name):
    def deco(fn):
        ACTIONS[name] = fn
        return fn
    return deco


def check_url(url):
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise ValueError("нужна ссылка вида https://...")
    return url


# ---------- кроссплатформенные медиа-клавиши и звук ----------
WIN_VK = {"mute": 0xAD, "down": 0xAE, "up": 0xAF, "next": 0xB0, "play": 0xB3}


def _win_key(vk, times=1):
    import ctypes
    u = ctypes.windll.user32
    for _ in range(times):
        u.keybd_event(vk, 0, 0, 0)
        u.keybd_event(vk, 0, 2, 0)


def _run(cmd):
    subprocess.run(cmd, check=True, timeout=10)


def _system(win_key, linux_cmd, mac_script=None, times=1):
    if sys.platform.startswith("win"):
        _win_key(WIN_VK[win_key], times)
    elif sys.platform == "darwin":
        if not mac_script:
            raise RuntimeError("на macOS это действие пока не поддерживается")
        _run(["osascript", "-e", mac_script])
    else:
        _run(linux_cmd)


def _browser_candidates():
    """Где обычно лежат браузеры. Возвращает {название: [пути/имена]}."""
    if sys.platform.startswith("win"):
        pf, pf86, loc = (os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"))
        j = os.path.join
        return {
            "brave": [j(p, "BraveSoftware", "Brave-Browser", "Application", "brave.exe") for p in (pf, pf86, loc)],
            "opera": [j(loc, "Programs", "Opera", "opera.exe"), j(loc, "Programs", "Opera GX", "opera.exe"),
                      j(pf, "Opera", "opera.exe"), j(pf86, "Opera", "opera.exe")],
            "chrome": [j(p, "Google", "Chrome", "Application", "chrome.exe") for p in (pf, pf86, loc)],
            "edge": [j(p, "Microsoft", "Edge", "Application", "msedge.exe") for p in (pf86, pf)],
            "firefox": [j(p, "Mozilla Firefox", "firefox.exe") for p in (pf, pf86)],
        }
    return {"brave": ["brave-browser", "brave"], "opera": ["opera"],
            "chrome": ["google-chrome", "chromium", "chromium-browser"],
            "edge": ["microsoft-edge"], "firefox": ["firefox"]}


_EXE = {"brave": "brave.exe", "opera": "opera.exe", "chrome": "chrome.exe", "edge": "msedge.exe",
        "firefox": "firefox.exe"}


def _registry_path(exe):
    """Windows: путь к программе из реестра («App Paths»), где её регистрирует установщик."""
    try:
        import winreg
    except ImportError:
        return None
    key = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, key) as k:
                p = winreg.QueryValueEx(k, "")[0].strip('"')
                if os.path.isfile(p):
                    return p
        except OSError:
            continue
    return None


def find_browsers():
    """Установленные браузеры: {название: путь}."""
    found = {}
    for name, paths in _browser_candidates().items():
        for p in paths:
            hit = p if os.path.isfile(p) else shutil.which(p)
            if hit:
                found[name] = hit
                break
        else:
            if sys.platform.startswith("win"):
                hit = _registry_path(_EXE[name])
                if hit:
                    found[name] = hit
    return found


def resolve_browser(value):
    """'brave' / 'opera' / путь к программе -> путь. Пустое значение = браузер по умолчанию (None)."""
    value = (value or "").strip()
    if not value:
        return None
    known = find_browsers()
    if value.lower() in known:
        return known[value.lower()]
    hit = value if os.path.isfile(value) else shutil.which(value)
    if not hit:
        raise FileNotFoundError(f"браузер не найден: {value} (список: browsers)")
    return hit


def _launch(browser, url):
    subprocess.Popen([browser, url])


def open_link(url, cmd, cfg):
    """Открывает ссылку в браузере: у команды свой (поле browser) важнее общего (browser в конфиге)."""
    url = check_url(url)
    browser = resolve_browser(cmd.get("browser") or cfg.get("browser"))
    if browser:
        _launch(browser, url)
        return f"открыта ссылка {url} ({os.path.basename(browser)})"
    if not webbrowser.open(url):
        raise RuntimeError("браузер не открылся")
    return f"открыта ссылка {url}"


@action("open_url")
def _open_url(cmd, cfg):
    url = cmd.get("target", "").strip()
    if not url:
        raise NeedTarget()
    return open_link(url, cmd, cfg)


@action("open_home")
def _open_home(cmd, cfg):
    url = (cmd.get("target") or cfg.get("home_url") or "").strip()
    if not url:
        raise NeedTarget()
    return open_link(url, cmd, cfg)


@action("open_app")
def _open_app(cmd, cfg):
    target = cmd.get("target", "").strip()
    if not target:
        raise NeedTarget()
    path = shutil.which(target) or (target if os.path.exists(target) else None)
    if not path:
        raise FileNotFoundError(f"не найдено: {target}")
    args = cmd.get("args", [])
    if sys.platform.startswith("win") and not args:
        os.startfile(path)
    else:
        subprocess.Popen([path, *args])
    return f"запущено {path}"


@action("volume_up")
def _vol_up(cmd, cfg):
    _system("up", ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"],
            "set volume output volume ((output volume of (get volume settings)) + 10)", times=5)
    return "громкость выше"


@action("volume_down")
def _vol_down(cmd, cfg):
    _system("down", ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"],
            "set volume output volume ((output volume of (get volume settings)) - 10)", times=5)
    return "громкость ниже"


@action("volume_mute")
def _vol_mute(cmd, cfg):
    _system("mute", ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
            "set volume output muted not (output muted of (get volume settings))")
    return "звук включён/выключен"


@action("media_play_pause")
def _play_pause(cmd, cfg):
    _system("play", ["playerctl", "play-pause"])
    return "пауза/продолжить"


@action("media_next")
def _next(cmd, cfg):
    _system("next", ["playerctl", "next"])
    return "следующий трек"


def _need_windows(what):
    if not sys.platform.startswith("win"):
        raise RuntimeError(f"«{what}» пока работает только на Windows")


def _win_combo(*vks):
    """Нажимает сочетание клавиш (например Win+D) и отпускает в обратном порядке."""
    import ctypes
    u = ctypes.windll.user32
    for vk in vks:
        u.keybd_event(vk, 0, 0, 0)
    for vk in reversed(vks):
        u.keybd_event(vk, 0, 2, 0)


VK_WIN, VK_ALT, VK_D, VK_F4, VK_SNAPSHOT, VK_MEDIA_PREV, VK_TAB = 0x5B, 0x12, 0x44, 0x73, 0x2C, 0xB1, 0x09

MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
          "ноября", "декабря"]
WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


def _open_path(path):
    """Открывает папку в проводнике (вынесено отдельно, чтобы проверять без настоящего проводника)."""
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


@action("open_folder")
def _open_folder(cmd, cfg):
    target = cmd.get("target", "").strip()
    if not target:
        raise NeedTarget()
    path = os.path.expanduser(target)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"папка не найдена: {path}")
    _open_path(path)
    return f"открыта папка {path}"


@action("tell_time")
def _tell_time(cmd, cfg):
    return Spoken(f"Сейчас {datetime.datetime.now():%H:%M}")


@action("tell_date")
def _tell_date(cmd, cfg):
    d = datetime.date.today()
    return Spoken(f"Сегодня {WEEKDAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]}")


@action("switch_window")
def _switch_window(cmd, cfg):
    _need_windows("переключить окно")
    _win_combo(VK_ALT, VK_TAB)
    return "окно переключено"


@action("media_prev")
def _prev(cmd, cfg):
    _system_prev()
    return "предыдущий трек"


def _system_prev():
    if sys.platform.startswith("win"):
        _win_key(VK_MEDIA_PREV)
    elif sys.platform == "darwin":
        raise RuntimeError("на macOS это действие пока не поддерживается")
    else:
        _run(["playerctl", "previous"])


@action("show_desktop")
def _show_desktop(cmd, cfg):
    _need_windows("свернуть все окна")
    _win_combo(VK_WIN, VK_D)
    return "окна свёрнуты"


@action("close_window")
def _close_window(cmd, cfg):
    _need_windows("закрыть окно")
    _win_combo(VK_ALT, VK_F4)
    return "окно закрыто"


@action("screenshot")
def _screenshot(cmd, cfg):
    _need_windows("скриншот")
    _win_combo(VK_WIN, VK_SNAPSHOT)      # Windows сама сохраняет в Изображения\Снимки экрана
    return "скриншот сохранён (Изображения → Снимки экрана)"


HOTKEY_KEYS = {"ctrl": 0x11, "shift": 0x10, "alt": 0x12, "win": 0x5B, "tab": 0x09, "enter": 0x0D, "esc": 0x1B,
               "space": 0x20, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "home": 0x24, "end": 0x23,
               "pageup": 0x21, "pagedown": 0x22, "delete": 0x2E, "backspace": 0x08, "printscreen": 0x2C,
               **{f"f{i}": 0x6F + i for i in range(1, 13)},
               **{ch: ord(ch.upper()) for ch in "abcdefghijklmnopqrstuvwxyz0123456789"}}


def parse_hotkey(spec):
    """'ctrl+shift+t' -> ['ctrl', 'shift', 't']. Только клавиши из списка, не больше 4."""
    keys = [k for k in (spec or "").lower().replace(" ", "").split("+") if k]
    bad = [k for k in keys if k not in HOTKEY_KEYS]
    if not keys or bad or len(keys) > 4:
        raise ValueError("сочетание вида ctrl+t, alt+left, f11 (до 4 клавиш"
                         + (f"; неизвестно: {', '.join(bad)}" if bad else "") + ")")
    return keys


@action("hotkey")
def _hotkey(cmd, cfg):
    """Нажимает сочетание клавиш (то, что вы задали в target): «новая вкладка» = ctrl+t и т. п."""
    target = (cmd.get("target") or "").strip()
    if not target:
        raise NeedTarget()
    keys = parse_hotkey(target)
    _need_windows("сочетание клавиш")
    _win_combo(*[HOTKEY_KEYS[k] for k in keys])
    return f"нажато {'+'.join(keys)}"


@action("screen_off")
def _screen_off(cmd, cfg):
    if sys.platform.startswith("win"):
        import ctypes
        ctypes.windll.user32.PostMessageW(0xFFFF, 0x0112, 0xF170, 2)     # монитор в спящий режим
    elif sys.platform == "darwin":
        _run(["pmset", "displaysleepnow"])
    else:
        _run(["xset", "dpms", "force", "off"])
    return "экран выключен"


@action("sequence")
def _sequence(cmd, cfg):
    """Сценарий: несколько команд подряд («рабочий режим» = браузер + музыка + ...)."""
    steps = cmd.get("steps") or []
    if not steps:
        raise RuntimeError("в сценарии нет шагов")
    done = []
    for i, name in enumerate(steps):
        c = cfg["commands"].get(name)
        if c is None or c.get("action") == "sequence":
            raise RuntimeError(f"шаг «{name}» недоступен")
        ok, status = execute(name, cfg)
        if not ok:
            if status == "no_target":
                raise NeedTarget()
            raise RuntimeError(f"шаг «{name}»: {status}")
        done.append(name)
        if i < len(steps) - 1:
            time.sleep(float(cmd.get("delay", 1.0)))      # чтобы окна открывались по порядку
    return "сценарий выполнен: " + ", ".join(done)


@action("lock_screen")
def _lock(cmd, cfg):
    if sys.platform.startswith("win"):
        import ctypes
        ctypes.windll.user32.LockWorkStation()
    elif sys.platform == "darwin":
        _run(["pmset", "displaysleepnow"])
    else:
        _run(["loginctl", "lock-session"])
    return "экран заблокирован"


def execute(name, cfg):
    """Выполняет команду по имени. Возвращает (успех, статус): 'no_target', 'error: ...' или описание."""
    cmd = cfg["commands"].get(name)
    if cmd is None:
        return False, "error: нет такой команды"
    fn = ACTIONS.get(cmd.get("action"))
    if fn is None:
        return False, f"error: неизвестное действие {cmd.get('action')}"
    try:
        return True, fn(cmd, cfg)
    except NeedTarget:
        return False, "no_target"
    except Exception as e:
        return False, f"error: {e}"


# Готовые команды: в терминале `presets` покажет список, `preset <имя>` добавит в конфиг.
PRESETS = {
    "screenshot": {"phrase": "сделай скриншот",   "action": "screenshot"},
    "desktop":    {"phrase": "сверни все окна",   "action": "show_desktop"},
    "close":      {"phrase": "закрой окно",       "action": "close_window"},
    "prev":       {"phrase": "предыдущий трек",   "action": "media_prev"},
    "youtube":    {"phrase": "открой ютуб",       "action": "open_url", "target": "https://www.youtube.com"},
    "calc":       {"phrase": "открой калькулятор", "action": "open_app", "target": "calc"},
    "notepad":    {"phrase": "открой блокнот",    "action": "open_app", "target": "notepad"},
    "taskmgr":    {"phrase": "диспетчер задач",   "action": "open_app", "target": "taskmgr"},
    "explorer":   {"phrase": "открой проводник",  "action": "open_app", "target": "explorer"},
    "downloads":  {"phrase": "открой загрузки",   "action": "open_folder", "target": "~/Downloads"},
    "time":       {"phrase": "который час",       "action": "tell_time"},
    "date":       {"phrase": "какое сегодня число", "action": "tell_date"},
    "switch":     {"phrase": "переключи окно",    "action": "switch_window"},
    "newtab":     {"phrase": "новая вкладка",     "action": "hotkey", "target": "ctrl+t"},
    "closetab":   {"phrase": "закрой вкладку",    "action": "hotkey", "target": "ctrl+w"},
    "refresh":    {"phrase": "обнови страницу",   "action": "hotkey", "target": "f5"},
    "fullscreen": {"phrase": "полный экран",      "action": "hotkey", "target": "f11"},
    "screen_off": {"phrase": "выключи экран",     "action": "screen_off"},
}
