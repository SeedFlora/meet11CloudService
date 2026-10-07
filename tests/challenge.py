"""Pemeriksaan end-to-end RAG lokal setelah Compose, ingest, dan Uvicorn berjalan."""

from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import nearest_rank  # noqa: E402
from rag import Embedder, extractive_answer, readiness  # noqa: E402

BASE_URL = "http://127.0.0.1:8011"
checks = 0


def check(label: str, condition: bool) -> None:
    global checks
    if not condition:
        raise AssertionError(label)
    checks += 1
    print("PASS", label)


def get(path: str) -> tuple[int, dict]:
    try:
        with urlopen(BASE_URL + path, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def ask(data: dict) -> tuple[int, dict]:
    request = Request(BASE_URL + "/api/ask", data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


model = Embedder()
check("embedding hash 384 dimensi", model.mode == "hash" and len(model.vector("docker container")) == 384)
check("chunk mempunyai awal dan akhir valid", all(chunk["end"] > chunk["start"] for chunk in model.chunks("a " * 600)))
check("p95 nearest rank konsisten", nearest_rank([1, 2, 3, 4, 5], .95) == 5)
status, health = get("/health")
check("liveness API HTTP 200", status == 200 and health["status"] == "ok")
status, ready = get("/ready")
check("readiness memeriksa indeks", status == 200 and ready["chunks"] > 0 and readiness()["chunks"] > 0)
status, result = ask({"question": "Apa perbedaan image dan container?", "top_k": 3})
check("pertanyaan menghasilkan jawaban dan sumber", status == 200 and len(result["sources"]) == 3 and bool(result["answer"]))
check("referensi sumber dapat dilacak", result["sources"][0]["ref"] == "S1" and result["sources"][0]["source_file"].endswith(".md"))
check("mode ekstraktif tidak memakai API berbayar", result["mode"] == "extractive")
status, invalid = ask({"question": "uji", "top_k": 0})
check("top_k nol ditolak HTTP 422", status == 422 and "detail" in invalid)
check("jawaban ekstraktif punya sitasi", "[S1]" in extractive_answer("Docker", [{"content": "Docker image berisi template untuk container.", "source_file": "x.md"}]))
print(f"challenge: {checks} PASS, 0 FAIL")
