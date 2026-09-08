"""The network door: the same three verbs, over HTTP, one memory per user.

A memory that lives in one process is a library. A product is asked from
somewhere else — a web app, a phone, a colleague's script — and until now
that meant every caller re-implemented the same wrapper around `Memory`,
each with its own idea of where a user's store lives and whether two
requests may touch it at once. Those are not application questions. They
are memory questions, and they are answered here.

Three things this module is careful about, because each is a way to break
the promise the rest of the codebase keeps:

EVERY USER'S MEMORY IS THEIR OWN. The store is chosen by the caller's
user id and nothing else, the id is checked to be a plain name before it
ever touches a path, and there is no route that reads across users. A
memory layer that leaked one tenant's documents into another tenant's
answer would be worse than useless, and "it is the application's job to
pass the right path" is how that leak gets written.

ONE WRITER AT A TIME. `Session` holds a graph and an evidence index and
mutates both while it learns; two requests learning into one store at
once is corruption, not concurrency. Each user's memory carries its own
lock, so different users still answer in parallel — which is the whole
reason for a threaded server — while one user's requests queue.

THE ANSWER IS THE SAME ANSWER. This is a transport, not a second brain:
it adds no gate, no rephrasing, no fallback text. What `Memory.ask`
returns — including its abstention and its sources — is what goes on the
wire, so a caller over HTTP can verify exactly what a caller in-process
can. `served` returns the same object the library does.

    python -m lmm.serve --root ./stores --port 8000
    curl -s localhost:8000/learn -d '{"user":"ada","text":"Ada leads R&D."}'
    curl -s localhost:8000/ask   -d '{"user":"ada","question":"who leads R&D?"}'

Binding is localhost by default: a memory full of a customer's documents
does not go on a public interface because a flag defaulted that way. Set
LMM_TOKEN to require `Authorization: Bearer <token>` on every request.
"""
import json
import os
import re
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lmm.api import Memory

# A USER ID IS A NAME, NOT A PATH. It arrives from the network and is used
# to build a filename, which is the classic way a store called "../../etc"
# reads a file nobody meant to share. Letters, digits, dash, underscore and
# dot — and never a leading dot or two dots alone, so no id can climb.
_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}$")

MAX_BODY = 4 << 20          # a request body is a question, not an upload


class Store:
    """The open memories, one per user, each with its own lock."""

    def __init__(self, root, **options):
        self.root = os.path.abspath(root)
        self.options = options
        self._open = {}
        self._guard = threading.Lock()

    def path_for(self, user):
        if not _ID.match(user or "") or user in (".", ".."):
            raise ValueError("bad user id")
        return os.path.join(self.root, user + ".lmm")

    def of(self, user):
        """(memory, lock) for this user — opened once, kept."""
        path = self.path_for(user)
        with self._guard:
            got = self._open.get(user)
            if got is None:
                os.makedirs(self.root, exist_ok=True)
                got = (Memory(path, **self.options), threading.RLock())
                self._open[user] = got
            return got

    def users(self):
        with self._guard:
            names = set(self._open)
        if os.path.isdir(self.root):
            names.update(f[:-4] for f in os.listdir(self.root)
                         if f.endswith(".lmm"))
        return sorted(names)


def _answer_json(answer):
    """The library's answer, unchanged, as JSON."""
    return {"answer": str(answer),
            "abstained": bool(getattr(answer, "abstained", False)),
            "sources": list(getattr(answer, "sources", ()) or ()),
            "subject": getattr(answer, "subject", "") or ""}


def handler_for(store, token=""):
    class Handler(BaseHTTPRequestHandler):
        server_version = "lmm"

        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("body too large")
            if not n:
                return {}
            got = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(got, dict):
                raise ValueError("body must be an object")
            return got

        def _allowed(self):
            if not token:
                return True
            return self.headers.get("Authorization", "") == "Bearer " + token

        def _user(self, body):
            return (body.get("user") or self.headers.get("X-LMM-User")
                    or "").strip()

        def log_message(self, *_a):
            pass                        # the request line carries user text

        def do_GET(self):
            if not self._allowed():
                return self._send(401, {"error": "unauthorized"})
            if self.path.split("?")[0] == "/health":
                return self._send(200, {"ok": True, "users": store.users()})
            self._send(404, {"error": "no such route"})

        def do_POST(self):
            if not self._allowed():
                return self._send(401, {"error": "unauthorized"})
            route = self.path.split("?")[0]
            try:
                body = self._body()
                user = self._user(body)
                memory, lock = store.of(user)
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:                       # noqa: BLE001
                return self._send(400, {"error": str(e)})
            try:
                with lock:
                    if route == "/ask":
                        q = body.get("question") or ""
                        if not q.strip():
                            return self._send(400, {"error": "no question"})
                        got = memory.ask(q, explain=True,
                                         fluent=bool(body.get("fluent")))
                        return self._send(200, _answer_json(got))
                    if route == "/learn":
                        what = body.get("text") or body.get("path") or ""
                        if not str(what).strip():
                            return self._send(400, {"error": "nothing to learn"})
                        got = memory.learn(what, source=body.get("source"),
                                           deep=body.get("deep"))
                        memory.save()
                        return self._send(200, {
                            "facts": got.facts, "evidence": got.evidence,
                            "source": got.source, "adapter": got.adapter,
                            "warnings": list(got.warnings)})
                    if route == "/compose":
                        brief = body.get("brief") or ""
                        if not brief.strip():
                            return self._send(400, {"error": "no brief"})
                        text, sources = memory.compose(brief)
                        return self._send(200, {"text": text,
                                                "sources": list(sources)})
                    if route == "/where":
                        term = body.get("term") or ""
                        return self._send(200, {"where": [
                            {"source": s, "hits": n}
                            for s, n in memory.where(term)]})
            except Exception as e:                       # noqa: BLE001
                return self._send(500, {"error": "%s: %s"
                                        % (type(e).__name__, e)})
            self._send(404, {"error": "no such route"})

    return Handler


def serve(root="stores", host="127.0.0.1", port=8000, token=None, **options):
    """Start the server and return it; caller runs serve_forever()."""
    store = Store(root, **options)
    token = os.environ.get("LMM_TOKEN", "") if token is None else token
    return ThreadingHTTPServer((host, port), handler_for(store, token))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="LMM over HTTP, one memory per user")
    p.add_argument("--root", default="stores", help="directory of user stores")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    httpd = serve(root=a.root, host=a.host, port=a.port)
    print("lmm serving on http://%s:%d  (stores in %s)"
          % (a.host, a.port, os.path.abspath(a.root)), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
