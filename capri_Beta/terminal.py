"""Мини-терминал Капри: задать цели команд (ссылки, программы), выбрать браузер, добавлять команды
и сценарии, проверить всё без голоса. Работает отдельно от слушателя: конфиг перечитывается при каждой команде.

  python terminal.py
"""
import cmd
import os
import re
import shlex
import shutil

import time

import actions
import learn
import tts
import voices
from common import CONFIG_PATH, SPECIAL, get_classes, load_config, new_command, save_config

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


def parse(arg):
    """Разбор аргументов; кавычки нужны для путей с пробелами (C:\\Program Files\\...)."""
    parts = shlex.split(arg, posix=False)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]


def split_phrase(parts, default):
    """Вынимает из аргументов «фраза=...» (если есть)."""
    phrase, rest = default, []
    for p in parts:
        if p.startswith("фраза="):
            phrase = p[len("фраза="):]
        else:
            rest.append(p)
    return phrase, rest


class CapriShell(cmd.Cmd):
    intro = "Мини-терминал Капри. Напишите help, чтобы увидеть команды."
    prompt = "capri> "

    def __init__(self, config_path=CONFIG_PATH, inbox=learn.INBOX, data_dir=learn.DATA):
        super().__init__()
        self.config_path, self.inbox, self.data_dir = config_path, inbox, data_dir

    def _cfg(self):
        return load_config(self.config_path)

    def _save(self, cfg):
        save_config(cfg, self.config_path)

    def emptyline(self):
        pass

    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except ValueError as e:
            print(f"Ошибка: {e}")

    def _new_name(self, cfg, name):
        if not NAME_RE.match(name) or name in SPECIAL or name in cfg["commands"]:
            raise ValueError("имя: латиница/цифры/_, начинается с буквы, не занято и не служебное")

    # ---------- просмотр ----------
    def do_list(self, arg):
        """list — все команды: что говорить, что делают, цели, ответы"""
        cfg = self._cfg()
        print(f"Имя: {cfg['name']}   браузер: {cfg['browser'] or 'по умолчанию'}   главная: {cfg['home_url']}")
        for name, c in cfg["commands"].items():
            if c["action"] == "sequence":
                what = "шаги: " + " → ".join(c.get("steps", []))
            elif c["action"] == "open_home":
                what = f"главная: {c.get('target') or cfg['home_url']}"
            else:
                what = f"цель: {c.get('target') or '—'}" + (f" {c['args']}" if c.get("args") else "")
            br = f"  браузер: {c['browser']}" if c.get("browser") else ""
            print(f"{name:12} «{c['phrase']}»  [{c['action']}]  {what}{br}")

    def do_actions(self, arg):
        """actions — какие действия можно назначать командам"""
        for name in sorted(actions.ACTIONS):
            need = actions.NEEDS_TARGET.get(name)
            print(f"{name:18} {'нужна цель: ' + need if need else ''}")

    # ---------- цели ----------
    def do_set(self, arg):
        """set <команда> <цель> — задать ссылку или программу. Пример: set music https://youtu.be/...
        Для программ можно добавить аргументы: set game "C:\\Games\\x.exe" --fullscreen"""
        parts = parse(arg)
        if len(parts) < 2:
            raise ValueError("используйте: set <команда> <цель>")
        name, target, extra = parts[0], parts[1], parts[2:]
        cfg = self._cfg()
        c = cfg["commands"].get(name)
        if c is None:
            raise ValueError(f"нет команды «{name}» (список: list)")
        if c["action"] == "open_url":
            if extra:
                raise ValueError("для ссылки лишние аргументы не нужны (ссылку с пробелами возьмите в кавычки)")
            c["target"] = actions.check_url(target)
        elif c["action"] == "open_app":
            if not (shutil.which(target) or os.path.exists(target)):
                raise ValueError(f"не найдено: {target}")
            c["target"], c["args"] = target, extra
        elif c["action"] == "open_folder":
            if extra or not os.path.isdir(os.path.expanduser(target)):
                raise ValueError(f"папка не найдена: {target} (путь с пробелами возьмите в кавычки)")
            c["target"] = target
        elif c["action"] == "hotkey":
            if extra:
                raise ValueError("сочетание пишется слитно: ctrl+shift+t")
            actions.parse_hotkey(target)
            c["target"] = target.lower()
        else:
            raise ValueError(f"у действия {c['action']} цель не нужна (главная страница: home, шаги: scene)")
        self._save(cfg)
        print(f"Готово: {name} → {target}")

    # ---------- браузер и главная ----------
    def do_browser(self, arg):
        """browser — какой браузер открывает ссылки и какие установлены
        browser <brave|opera|chrome|edge|firefox|путь> — выбрать для всех команд
        browser <имя> <команда> — свой браузер только для одной команды (например музыка в Brave)
        browser default [команда] — вернуть браузер по умолчанию"""
        parts = parse(arg)
        cfg = self._cfg()
        if not parts:
            found = actions.find_browsers()
            print(f"Сейчас: {cfg['browser'] or 'по умолчанию (системный)'}")
            for n, c in cfg["commands"].items():
                if c.get("browser"):
                    print(f"  у команды {n}: {c['browser']}")
            print("Найдены на этом ПК: " + (", ".join(f"{k} ({v})" for k, v in found.items()) or "ничего"))
            return
        value, only = parts[0], (parts[1] if len(parts) > 1 else None)
        if value.lower() in ("default", "system"):
            new = ""
        else:
            try:
                actions.resolve_browser(value)
            except FileNotFoundError as e:
                raise ValueError(str(e))
            new = value.lower() if value.lower() in actions.find_browsers() else value
        if only:
            c = cfg["commands"].get(only)
            if c is None or c["action"] not in ("open_url", "open_home"):
                raise ValueError(f"«{only}» не открывает ссылки (команды со ссылками: list)")
            if new:
                c["browser"] = new
            else:
                c.pop("browser", None)
        else:
            cfg["browser"] = new
        self._save(cfg)
        print(f"Готово: {(only or 'все команды')} → {new or 'браузер по умолчанию'}")

    def do_home(self, arg):
        """home — показать главную страницу; home <ссылка> — задать (её открывает «открой браузер»)"""
        cfg = self._cfg()
        arg = arg.strip()
        if not arg:
            print(f"Главная: {cfg['home_url']}")
            return
        cfg["home_url"] = actions.check_url(parse(arg)[0])
        self._save(cfg)
        print(f"Готово: главная → {cfg['home_url']}")

    # ---------- окно на рабочем столе ----------
    def do_hud(self, arg):
        """hud — настройки окна Капри на рабочем столе
        hud on|off — включить / выключить окно (действует после перезапуска main.py)
        hud corner <top_left|top_right|bottom_left|bottom_right> — в каком углу экрана
        Перетащить окно — левой кнопкой мыши, закрыть — правой (Выход)"""
        parts = parse(arg)
        cfg = self._cfg()
        h = cfg["hud"]
        if not parts:
            print(f"Окно: {'включено' if h['enabled'] else 'выключено'}   угол: {h['corner']}")
            return
        key = parts[0].lower()
        if key in ("on", "off") and len(parts) == 1:
            h["enabled"] = key == "on"
        elif key == "corner" and len(parts) == 2 and parts[1] in ("top_left", "top_right", "bottom_left", "bottom_right"):
            h["corner"] = parts[1]
        else:
            raise ValueError("используйте: hud on|off, hud corner <угол>")
        self._save(cfg)
        print(f"Готово: окно {'включено' if h['enabled'] else 'выключено'}, угол {h['corner']}")

    # ---------- ответы и фразы ----------
    def do_reply(self, arg):
        """reply <команда> <текст> — что отвечает Капри после выполнения (голос — файл voices/<команда>.wav, иначе системный)"""
        parts = parse(arg)
        if len(parts) < 2:
            raise ValueError("используйте: reply <команда> <текст>")
        cfg = self._cfg()
        c = cfg["commands"].get(parts[0])
        if c is None:
            raise ValueError(f"нет команды «{parts[0]}»")
        c["reply"] = " ".join(parts[1:])
        c["voice"] = "ok" if c["reply"] == cfg["replies"]["ok"] else parts[0]
        self._save(cfg)
        print("Готово" + ("" if c["voice"] == "ok" else f". Свой голос для этой фразы: voices/{parts[0]}.wav"))

    def do_phrase(self, arg):
        """phrase <команда> <фраза> — что вы говорите (подсказка для записи голоса)"""
        parts = parse(arg)
        if len(parts) < 2:
            raise ValueError("используйте: phrase <команда> <фраза>")
        cfg = self._cfg()
        if parts[0] not in cfg["commands"]:
            raise ValueError(f"нет команды «{parts[0]}»")
        cfg["commands"][parts[0]]["phrase"] = " ".join(parts[1:])
        self._save(cfg)
        print("Готово")

    def do_voices(self, arg):
        """voices — какие фразы должна произносить Капри и каких файлов voices/*.wav ещё нет"""
        print(voices.report(self._cfg()))

    # ---------- новые команды ----------
    def do_add(self, arg):
        """add <имя> <действие> [цель] [фраза=...] — новая команда. Пример: add rock open_url https://...
        Для каждой ссылки на музыку можно завести отдельную команду со своей фразой."""
        parts = parse(arg)
        if len(parts) < 2:
            raise ValueError("используйте: add <имя> <действие> [цель]  (действия: actions)")
        name, act = parts[0], parts[1]
        phrase, rest = split_phrase(parts[2:], name.replace("_", " "))
        cfg = self._cfg()
        self._new_name(cfg, name)
        if act not in actions.ACTIONS:
            raise ValueError("неизвестное действие (смотрите: actions)")
        if act == "sequence":
            raise ValueError("сценарии создаются командой scene")
        cfg["commands"][name] = new_command(cfg, phrase, act)
        self._save(cfg)
        if act in actions.NEEDS_TARGET:
            if not rest:
                print(f"Команда добавлена, цель не задана: set {name} <цель>")
            else:
                self.onecmd(f"set {name} " + " ".join(shlex.quote(x) if " " in x else x for x in rest))
        print(f"Добавлена команда «{name}». Теперь запишите голос (python record_data.py {name}) "
              f"и доучите: python train.py --grow")

    def do_presets(self, arg):
        """presets — готовые команды, которые можно добавить одной строкой"""
        have = self._cfg()["commands"]
        for name, p in actions.PRESETS.items():
            print(f"{name:12} «{p['phrase']}»  [{p['action']}]" + ("   (уже добавлена)" if name in have else ""))
        print("Добавить: preset <имя>")

    def do_preset(self, arg):
        """preset <имя> [имя ...] — добавить готовые команды (список: presets). Лучше по 3–5 за раз."""
        names = arg.split()
        if not names:
            raise ValueError("используйте: preset <имя> [имя ...]  (список: presets)")
        cfg = self._cfg()
        added = []
        for name in names:
            p = actions.PRESETS.get(name)
            if p is None:
                print(f"  «{name}»: нет такого набора (список: presets)")
            elif name in cfg["commands"]:
                print(f"  «{name}»: уже добавлена")
            else:
                cfg["commands"][name] = new_command(cfg, p["phrase"], p["action"], p.get("target", ""))
                added.append(name)
                print(f"  добавлено: «{p['phrase']}»")
        if added:
            self._save(cfg)
            print(f"Дальше: python record_data.py   (запишет только то, чего не хватает), затем python train.py --grow")

    def do_scene(self, arg):
        """scene <имя> <команда1> <команда2> ... [фраза=...] — сценарий: несколько команд одной фразой.
        Пример: scene work browser music "фраза=рабочий режим"  (откроет главную и включит музыку)"""
        parts = parse(arg)
        if len(parts) < 3:
            raise ValueError("используйте: scene <имя> <команда1> <команда2> ...")
        name = parts[0]
        phrase, steps = split_phrase(parts[1:], name.replace("_", " "))
        cfg = self._cfg()
        self._new_name(cfg, name)
        for s in steps:
            c = cfg["commands"].get(s)
            if c is None:
                raise ValueError(f"нет команды «{s}» (список: list)")
            if c["action"] == "sequence":
                raise ValueError("сценарий внутри сценария не поддерживается")
        cfg["commands"][name] = new_command(cfg, phrase, "sequence", steps=steps, delay=1.0)
        self._save(cfg)
        print(f"Сценарий «{name}»: {' → '.join(steps)}. Запишите голос: python record_data.py {name}  "
              f"и доучите: python train.py --grow")

    def do_remove(self, arg):
        """remove <имя> — удалить команду (потребуется переобучение)"""
        name = arg.strip()
        cfg = self._cfg()
        if name not in cfg["commands"]:
            raise ValueError(f"нет команды «{name}»")
        used = [n for n, c in cfg["commands"].items() if name in c.get("steps", [])]
        if used:
            raise ValueError(f"«{name}» входит в сценарий {', '.join(used)}: сначала удалите его")
        if input(f"Удалить «{name}»? (да/нет) ").strip().lower() not in ("да", "y", "yes"):
            return print("Отменено")
        del cfg["commands"][name]
        self._save(cfg)
        print("Удалено. Перенесите выученное в новый список: python train.py --grow")

    # ---------- проверка без голоса ----------
    def do_run(self, arg):
        """run <команда> — выполнить команду прямо сейчас (проверка без голоса)"""
        cfg = self._cfg()
        name = arg.strip()
        ok, status = actions.execute(name, cfg)
        print(("Выполнено: " if ok else "Не вышло: ") + status)
        c = cfg["commands"].get(name)
        if c:
            if ok:
                if isinstance(status, actions.Spoken):
                    tts.say(str(status))
                else:
                    tts.say(c["reply"], c.get("voice"))
            else:
                key = "no_target" if status == "no_target" else "error"
                tts.say(cfg["replies"][key], key)

    def do_say(self, arg):
        """say <текст> — проверить голос"""
        tts.say(arg.strip())

    # ---------- обучение на опыте ----------
    def _show_inbox(self, limit=12):
        clips = learn.list_clips(self.inbox)
        if not clips:
            return print("Inbox пуст: сюда попадают фразы, на которые Капри проснулась, выполнила команду или не поняла.")
        print(f"В inbox {len(clips)} (показаны новые):")
        for c in clips[:limit]:
            print(f"  {c['n']:>3}. {time.strftime('%d.%m %H:%M:%S', time.localtime(c['time']))}  "
                  f"поняла как «{c['pred']}» ({c['conf']:.0%}, {c['event']})")

    def do_inbox(self, arg):
        """inbox — что Капри недавно слышала (нужно проверить: поняла ли правильно)
        Дальше: play <№> послушать · fix <№> [команда] отнести к команде (без команды — как поняла) · drop <№> удалить"""
        self._show_inbox()

    def do_play(self, arg):
        """play <№> — проиграть запись из inbox"""
        clip = learn.get_clip(self._num(arg), self.inbox)
        try:
            tts._play_wav(clip["path"])
        except Exception as e:
            print(f"Не удалось проиграть: {e}")

    def do_fix(self, arg):
        """fix <№> [команда] — запись становится примером этой команды (без команды — как Капри поняла;
        _unknown — «это что-то другое», _silence — шум). Затем: python train.py --more 20"""
        parts = parse(arg)
        if not parts or len(parts) > 2:
            raise ValueError("используйте: fix <№> [команда]")
        clip = learn.get_clip(self._num(parts[0]), self.inbox)
        cls = parts[1] if len(parts) == 2 else clip["pred"]
        if cls not in get_classes(self._cfg()):
            raise ValueError(f"нет такого класса «{cls}» (команды: list; ещё _unknown, _silence, wake)")
        dest = learn.move_clip(clip, cls, self.data_dir)
        print(f"Записано в обучение: {os.path.relpath(dest, self.data_dir)}. Дообучить: python train.py --more 20")
        self._show_inbox()

    def do_drop(self, arg):
        """drop <№> — удалить запись из inbox (не годится для обучения)"""
        learn.drop_clip(learn.get_clip(self._num(arg), self.inbox))
        self._show_inbox()

    @staticmethod
    def _num(s):
        try:
            return int(str(s).strip())
        except ValueError:
            raise ValueError("нужен номер записи из inbox")

    def do_exit(self, arg):
        """exit — выйти"""
        return True

    do_quit = do_exit

    def do_EOF(self, arg):
        print()
        return True



if __name__ == "__main__":
    CapriShell().cmdloop()
