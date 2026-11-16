"""Синхронизация обучения между ПК (без внешних библиотек, только stdlib + numpy).

  Главный ПК:      python federated.py server --port 8765
  Остальные ПК:    python train.py --server http://IP_ГЛАВНОГО:8765 --token ТОКЕН

Как это работает: каждый ПК после чекпоинта присылает свои веса и число обучающих примеров.
Сервер хранит ПОСЛЕДНИЙ вклад каждого ПК и усредняет их, взвешивая по числу примеров (FedAvg).
Если ПК выключен или нет сети, он просто учится локально и досылает позже.

Безопасность: токен обязателен; файлы читаются с allow_pickle=False, форма и значения весов
проверяются. Передача идёт по обычному HTTP: используйте домашнюю сеть или VPN (WireGuard/Tailscale).
"""
import argparse
import hmac
import io
import json
import os
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from common import (CONFIG_PATH, atomic_savez, atomic_write_bytes, build_model, classes_hash,
                    get_classes, load_config, read_npz)

MAX_BODY = 16 * 1024 * 1024
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9_-]", "-", name)[:32]
    return name or "pc"


def pack_weights(weights):
    buf = io.BytesIO()
    np.savez(buf, **weights)
    return buf.getvalue()


def parse_weights(data, ref_shapes):
    """Безопасно читает веса из байтов; бросает ValueError, если что-то не так."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():                       # защита от «раздутых» архивов
                key = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
                if key not in ref_shapes:
                    raise ValueError(f"лишний массив {info.filename}")
                if info.file_size > int(np.prod(ref_shapes[key])) * 8 + 1024:
                    raise ValueError(f"массив {key} слишком большой")
        out = {}
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            if set(z.files) != set(ref_shapes):
                raise ValueError("набор массивов не совпадает с моделью")
            for k, shape in ref_shapes.items():
                a = z[k]
                if a.shape != tuple(shape) or a.dtype.kind != "f":
                    raise ValueError(f"массив {k}: неверная форма или тип")
                if not np.isfinite(a).all():
                    raise ValueError(f"массив {k}: NaN/Inf")
                out[k] = a.astype(np.float32)
        return out
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"не удалось прочитать веса: {e}")


# ======================= сервер =======================
class Aggregator:
    def __init__(self, state_dir, classes):
        self.dir = state_dir
        self.chash = classes_hash(classes)
        self.lock = threading.Lock()
        init = build_model(len(classes)).weights_dict()
        self.shapes = {k: v.shape for k, v in init.items()}
        self.slots = {}                  # имя ПК -> {"w":..., "n":..., "time":...}
        self.round = 0
        wdir = os.path.join(state_dir, "workers")
        os.makedirs(wdir, exist_ok=True)
        for fn in sorted(os.listdir(wdir)):
            if fn.endswith(".npz") and NAME_RE.match(fn[:-4]):
                try:
                    data, meta = read_npz(os.path.join(wdir, fn))
                    self.slots[fn[:-4]] = {"w": parse_weights(pack_weights(data), self.shapes),
                                           "n": int(meta["n"]), "time": float(meta["time"])}
                except Exception as e:
                    print(f"[server] пропускаю {fn}: {e}")
        gpath = os.path.join(state_dir, "global.npz")
        if os.path.exists(gpath):
            try:
                self.round = int(read_npz(gpath)[1].get("round", 0))
            except Exception:
                pass
        self.global_w = self._average() or init

    def _average(self):
        if not self.slots:
            return None
        tot = sum(s["n"] for s in self.slots.values())
        return {k: sum(s["n"] * s["w"][k] for s in self.slots.values()) / tot for k in self.shapes}

    def update(self, name, weights, n):
        with self.lock:
            now = time.time()
            self.slots[name] = {"w": weights, "n": max(1, n), "time": now}
            atomic_savez(os.path.join(self.dir, "workers", name + ".npz"), weights, {"n": max(1, n), "time": now})
            self.global_w = self._average()
            self.round += 1
            atomic_savez(os.path.join(self.dir, "global.npz"), self.global_w,
                         {"round": self.round, "chash": self.chash})
            return self.round, len(self.slots)

    def snapshot(self):
        with self.lock:
            return pack_weights(self.global_w), self.round

    def status(self):
        with self.lock:
            now = time.time()
            return {"round": self.round, "classes_hash": self.chash,
                    "workers": {k: {"samples": s["n"], "seconds_ago": int(now - s["time"])}
                                for k, s in self.slots.items()}}


class Handler(BaseHTTPRequestHandler):
    server_version = "CapriFed/1"
    timeout = 20

    def log_message(self, fmt, *args):
        print(f"[server] {self.address_string()} {fmt % args}")

    def _send(self, code, body=b"", ctype="application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj, headers=None):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), headers=headers)

    def _authorized(self):
        got = self.headers.get("X-Token", "").encode("utf-8", "replace")
        if hmac.compare_digest(got, self.server.token.encode("utf-8")):
            return True
        self._json(403, {"error": "неверный токен"})
        return False

    def _classes_ok(self):
        if self.headers.get("X-Classes", "") == self.server.agg.chash:
            return True
        self._json(409, {"error": "список команд на этом ПК не совпадает с главным"})
        return False

    def do_GET(self):
        if not self._authorized():
            return
        if self.path == "/status":
            return self._json(200, self.server.agg.status())
        if self.path == "/model":
            if not self._classes_ok():
                return
            body, rnd = self.server.agg.snapshot()
            return self._send(200, body, "application/octet-stream", {"X-Round": str(rnd)})
        self._json(404, {"error": "нет такого пути"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_BODY:
            self.close_connection = True
            return self._json(413 if length > MAX_BODY else 400, {"error": "неверный размер запроса"})
        body = self.rfile.read(length)      # тело дочитываем ВСЕГДА, иначе клиент увидит обрыв вместо ответа
        if not self._authorized():
            return
        if self.path != "/update":
            return self._json(404, {"error": "нет такого пути"})
        if not self._classes_ok():
            return
        name = self.headers.get("X-Worker", "")
        if not NAME_RE.match(name):
            return self._json(400, {"error": "имя ПК: латиница, цифры, - и _, до 32 символов"})
        try:
            n = int(self.headers.get("X-Samples", "1"))
        except ValueError:
            return self._json(400, {"error": "неверные заголовки"})
        try:
            w = parse_weights(body, self.server.agg.shapes)
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        rnd, workers = self.server.agg.update(name, w, n)
        self._json(200, {"round": rnd, "workers": workers})


def make_server(state_dir, classes, token, host="0.0.0.0", port=8765):
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.agg = Aggregator(state_dir, classes)
    srv.token = token
    return srv


# ======================= клиент =======================
class FederatedClient:
    def __init__(self, url, token, worker=None, timeout=15):
        self.url = url.rstrip("/")
        self.token = token
        self.worker = safe_name(worker or socket.gethostname())
        self.timeout = timeout

    def _call(self, method, path, chash, data=None, extra=None):
        h = {"X-Token": self.token, "X-Classes": chash, "X-Worker": self.worker}
        h.update(extra or {})
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read(), r.headers
        except urllib.error.HTTPError as e:         # показываем понятную причину от сервера
            try:
                msg = json.loads(e.read()).get("error", "")
            except Exception:
                msg = ""
            raise RuntimeError(f"сервер ответил {e.code}: {msg}") from None

    def push(self, weights, n_samples, chash):
        """Отправляет веса. Возвращает ответ сервера или None (сервер недоступен — не страшно)."""
        try:
            body, _ = self._call("POST", "/update", chash, pack_weights(weights),
                                 {"X-Samples": str(n_samples), "Content-Type": "application/octet-stream"})
            return json.loads(body)
        except Exception as e:
            print(f"[sync] отправка не удалась: {e}")
            return None

    def pull(self, ref_weights, chash):
        """Скачивает общую модель. Возвращает (веса, раунд) или None."""
        try:
            body, hdr = self._call("GET", "/model", chash)
            shapes = {k: v.shape for k, v in ref_weights.items()}
            return parse_weights(body, shapes), int(hdr.get("X-Round", 0))
        except Exception as e:
            print(f"[sync] загрузка не удалась: {e}")
            return None


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("server", help="запустить сервер на главном ПК")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token", default=None)
    s.add_argument("--state", default="server_state")
    s.add_argument("--config", default=CONFIG_PATH)
    a = ap.parse_args()

    classes = get_classes(load_config(a.config))
    tpath = os.path.join(a.state, "token.txt")
    token = a.token
    if not token:
        if os.path.exists(tpath):
            token = open(tpath, encoding="utf-8").read().strip()
        else:
            token = secrets.token_urlsafe(16)
            atomic_write_bytes(tpath, token.encode("utf-8"))
    srv = make_server(a.state, classes, token, a.host, a.port)
    print(f"Сервер слушает {a.host}:{a.port}, раунд {srv.agg.round}, ПК в базе: {list(srv.agg.slots)}")
    print(f"Токен: {token}")
    print(f"На других ПК: python train.py --server http://<IP-этого-ПК>:{a.port} --token {token}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено")


if __name__ == "__main__":
    main()
