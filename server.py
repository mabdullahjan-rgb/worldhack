import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ARENA = Path(__file__).with_name("arena.html")
MODEL_URL = "http://127.0.0.1:11434/api/chat"
ROLES = ["volcano", "water", "shockwave", "speed"]
FIELDS = {
    "name": {"type": "string"},
    "title": {"type": "string"},
    "feature": {"type": "string"},
    "role": {"type": "string", "enum": ROLES},
    "box": {
        "type": "array", "minItems": 4, "maxItems": 4,
        "items": {"type": "number", "minimum": 0, "maximum": 1000},
    },
}
FORMAT = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array", "maxItems": 5,
            "items": {
                "type": "object", "properties": FIELDS,
                "required": list(FIELDS), "additionalProperties": False,
            },
        },
    },
    "required": ["objects"], "additionalProperties": False,
}
PROMPT = """
Turn this photo into an escape-game level. Identify up to five distinct,
clearly visible foreground objects. Ignore people and background clutter.
Return fewer objects if needed. Never describe fictional features as real.
For each object, give its real name, a short creative title, one visible
feature that inspired the mechanic, and its approximate bounding box as
[left, top, right, bottom], each on a 0 to 1000 scale across the full photo.

Choose one supported role per object using playful analogies. Roles:
volcano = a timed eruption hazard; water = a charge to cool an eruption;
shockwave = a charge to cancel an eruption; speed = a temporary sprint.
Use volcano and water at most once each. Choose varied roles when they fit.
Keep invented names in title, and put only a supported role ID in role.
Return only the requested JSON.
"""


class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, payload):
        self.send_data(status, "application/json", json.dumps(payload).encode())

    def do_GET(self):
        if self.path in {"/", "/arena.html"}:
            self.send_data(200, "text/html; charset=utf-8", ARENA.read_bytes())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/scan":
            self.send_error(404)
            return

        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= 4_000_000 or self.headers.get_content_type() != "image/jpeg":
            self.send_json(400, {"error": "Send a JPEG under 4 MB."})
            return

        image = base64.b64encode(self.rfile.read(size)).decode("ascii")
        payload = {
            "model": "gemma3:4b",
            "messages": [{"role": "user", "content": PROMPT, "images": [image]}],
            "format": FORMAT,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        request = Request(MODEL_URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=180) as response:
                result = json.load(response)
            parsed = json.loads(result["message"]["content"])
            if not isinstance(parsed, dict) or not isinstance(parsed.get("objects"), list):
                raise ValueError("Invalid model result")
        except HTTPError as error:
            self.send_json(502, {"error": f"Ollama returned HTTP {error.code}. Check that gemma3:4b is installed."})
            return
        except (URLError, TimeoutError):
            self.send_json(502, {"error": "Ollama did not respond. Open Ollama and try again."})
            return
        except (ValueError, KeyError, TypeError):
            self.send_json(502, {"error": "Gemma returned an unreadable result. Try another photo."})
            return
        self.send_json(200, parsed)


if __name__ == "__main__":
    address = ("127.0.0.1", 8000)
    print("WORLDHACK is running at http://localhost:8000", flush=True)
    with ThreadingHTTPServer(address, Handler) as server:
        server.serve_forever()
