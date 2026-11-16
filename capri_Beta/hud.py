"""Капри на рабочем столе: окно без рамки с анимированным персонажем (avatar/capri.gif) поверх всех окон.
Фон прозрачный (Windows). Левая кнопка мыши — перетащить, правая — меню «Выход».

  HudModel   — состояние (проснулась / говорит), без tkinter: проверяется тестами
  HudWindow  — тонкая обёртка над tkinter
Свою анимацию: положите gif в папку avatar/ и впишите имя в config.json -> hud.image.
"""
import os
import queue
import sys
import time
import traceback

KEY = "#010203"                # этот цвет Windows делает прозрачным (он же — прозрачный цвет в gif)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HUD = {"enabled": True, "corner": "bottom_right", "click_through": False, "image": "capri.gif"}
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


class HudModel:
    def __init__(self, load_cfg, clock=time.monotonic):
        self.load_cfg, self.clock = load_cfg, clock
        self.events = queue.Queue()
        self.awake_until = 0.0
        self.wake_seconds = 6.0
        self.speaking = False
        self.quit = False
        self.cfg_hud = dict(DEFAULT_HUD)
        self._cfg_at = -1e9

    def post(self, event, **info):                 # из любого потока
        self.events.put((event, info))

    def set_level(self, rms):                      # уровень микрофона больше не рисуется; метод оставлен для слушателя
        pass

    def pump(self):
        while True:
            try:
                event, info = self.events.get_nowait()
            except queue.Empty:
                return
            self.apply(event, info)

    def apply(self, event, info):
        now = self.clock()
        if event == "wake":
            self.wake_seconds = float(info.get("seconds", 6.0))
            self.awake_until = now + self.wake_seconds
        elif event == "command":
            self.awake_until = 0.0
        elif event == "speak":
            self.speaking = True
        elif event == "speak_end":
            self.speaking = False
            if self.awake_until > now:
                self.awake_until = now + self.wake_seconds
        elif event == "quit":
            self.quit = True

    def active(self, now):
        """Капри слушает или говорит: оживляется быстрее и становится непрозрачной."""
        return self.speaking or now < self.awake_until

    def speed(self, now):
        return 1.6 if self.active(now) else 1.0

    def alpha(self, now):
        return 1.0 if self.active(now) else 0.82

    def refresh_cfg(self):
        now = self.clock()
        if now - self._cfg_at > 1.5:               # настройки перечитываем раз в 1.5 с
            self._cfg_at = now
            try:
                self.cfg_hud = {**DEFAULT_HUD, **self.load_cfg().get("hud", {})}
            except Exception:
                pass


def _work_area(root):
    """Рабочая область экрана без панели задач (Windows) или весь экран."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes
            r = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(r), 0):     # SPI_GETWORKAREA
                return r.left, r.top, r.right, r.bottom
        except Exception:
            pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


class HudWindow:
    def __init__(self, model, load_cfg, on_quit=None):
        import tkinter as tk                       # внутри: без tkinter остальная программа работает как раньше
        self._tk = tk
        self.m, self.on_quit = model, on_quit
        self.root = tk.Tk()
        self.root.title("Capri")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", KEY)      # только Windows
        except tk.TclError:
            pass
        self.root.configure(bg=KEY)
        try:
            self.m.cfg_hud = {**DEFAULT_HUD, **load_cfg().get("hud", {})}
        except Exception:
            pass
        self.frames = self._load_frames(self.m.cfg_hud.get("image"))
        if not self.frames:
            raise RuntimeError("не найден gif персонажа (avatar/capri.gif)")
        self.W, self.H = self.frames[0].width(), self.frames[0].height()
        self.c = tk.Canvas(self.root, width=self.W, height=self.H, bg=KEY, highlightthickness=0, bd=0)
        self.c.pack()
        self.item = self.c.create_image(0, 0, image=self.frames[0], anchor="nw")
        self.root.report_callback_exception = self._on_exception
        self._corner, self._visible, self._closed = None, True, False
        self._i, self._acc, self._last = 0, 0.0, self.m.clock()
        self._place(self.m.cfg_hud["corner"])
        self.root.update_idletasks()
        self._win_flags(bool(self.m.cfg_hud.get("click_through")))
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Выход", command=self._shutdown)
        self.menu = menu
        self.c.bind("<Button-3>", self._popup)
        self.c.bind("<ButtonPress-1>", self._drag_start)
        self.c.bind("<B1-Motion>", self._drag)

    def _load_frames(self, name):
        path = os.path.join(HERE, "avatar", os.path.basename(str(name or "capri.gif")))
        frames = []
        while True:
            try:
                frames.append(self._tk.PhotoImage(file=path, format=f"gif -index {len(frames)}"))
            except self._tk.TclError:
                break
        return frames

    def _place(self, corner):
        l, t, r, b = _work_area(self.root)
        mg = 8
        x = l + mg if "left" in corner else r - self.W - mg
        y = t + mg if "top" in corner else b - self.H - mg
        self.root.geometry(f"{self.W}x{self.H}+{x}+{y}")
        self._corner = corner

    def _drag_start(self, e):
        self._dx, self._dy = e.x, e.y

    def _drag(self, e):
        self.root.geometry(f"+{self.root.winfo_x() + e.x - self._dx}+{self.root.winfo_y() + e.y - self._dy}")

    def _win_flags(self, click_through):
        """Windows: окно не забирает фокус, не в панели задач; по желанию пропускает клики мышки."""
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes
            u = ctypes.windll.user32
            hwnd = u.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            st = u.GetWindowLongW(hwnd, -20)                    # GWL_EXSTYLE
            st |= 0x08000000 | 0x80                             # NOACTIVATE | TOOLWINDOW
            if click_through:
                st |= 0x20 | 0x80000                            # TRANSPARENT | LAYERED
            u.SetWindowLongW(hwnd, -20, st)
        except Exception:
            pass

    def _popup(self, event):
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _on_exception(self, exc, val, tb):
        if issubclass(exc, KeyboardInterrupt):
            self._shutdown()
        else:
            traceback.print_exception(exc, val, tb)

    def _shutdown(self):
        if self._closed:
            return
        self._closed = True
        if self.on_quit:
            self.on_quit()
        try:
            self.root.destroy()
        except Exception:
            pass

    def _tick(self):
        if self._closed:
            return
        if self.m.quit:
            return self._shutdown()
        now = self.m.clock()
        dt = min(0.25, now - self._last)
        self._last = now
        self.m.pump()
        self.m.refresh_cfg()
        h = self.m.cfg_hud
        if h.get("corner") != self._corner and h.get("corner") in CORNERS:
            self._place(h["corner"])
        visible = bool(h.get("enabled", True))
        if visible != self._visible:
            self._visible = visible
            (self.root.deiconify if visible else self.root.withdraw)()
        if visible:
            self._acc += dt * self.m.speed(now)
            while self._acc >= 0.045:                           # кадр gif — 45 мс
                self._acc -= 0.045
                self._i = (self._i + 1) % len(self.frames)
            self.c.itemconfigure(self.item, image=self.frames[self._i])
            try:
                self.root.attributes("-alpha", self.m.alpha(now))
            except Exception:
                pass
        self.root.after(40, self._tick)

    def run(self):
        self.root.after(40, self._tick)
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()
