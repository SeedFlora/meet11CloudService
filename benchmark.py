"""Ukur p50/p95 dari beberapa pengulangan pertanyaan RAG kelas."""

import argparse
from math import ceil
import json
from pathlib import Path
from statistics import median

from rag import ROOT, answer_question

QUESTIONS = [
    "Apa perbedaan image dan container?",
    "Mengapa Row Level Security penting untuk Supabase?",
    "Bagaimana healthcheck dan volume membantu Docker Compose?",
]


def nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5, help="pengulangan tiap pertanyaan")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks.jsonl")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat minimal 1")
    latencies: list[float] = []
    with args.output.open("w", encoding="utf-8") as stream:
        for question in QUESTIONS:
            for repetition in range(1, args.repeat + 1):
                result = answer_question(question)
                row = {"question": question, "repetition": repetition, **result}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                latencies.append(result["latency_ms"])
                print(f'{result["latency_ms"]:8.2f} ms | sumber: {len(result["sources"])} | {question}')
    print(f"Sampel: {len(latencies)}; p50={median(latencies):.2f} ms; p95={nearest_rank(latencies, .95):.2f} ms")
    print(f"Hasil: {args.output}")


if __name__ == "__main__":
    main()
