"""
Interactive terminal interface for the Healthcare Information Assistant.

"""
from __future__ import annotations

import argparse

from src.pipeline import HealthcareRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Healthcare Information Assistant CLI")
    parser.add_argument("--embedder", default="tfidf", choices=["tfidf", "st"])
    parser.add_argument("--generator", default="extractive", choices=["extractive", "groq"])
    args = parser.parse_args()

    print("Building knowledge base...")
    pipeline = HealthcareRAGPipeline(
        embedding_backend=args.embedder,
        generator_backend=args.generator,
    )
    pipeline.build()
    print(f"Loaded {pipeline.num_documents} documents -> {pipeline.num_chunks} chunks.\n")
    print("Healthcare Information Assistant. Type 'exit' to quit.\n")
    print("(This provides general information from cited sources, not medical advice.)\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response = pipeline.answer(query)
        print(f"\nAssistant: {response.answer}\n")
        if response.confidence:
            print(f"Confidence: {response.confidence.label} ({response.confidence.score}) "
                  f"— {response.confidence.rationale}")
        if response.citations:
            print("Sources:")
            for c in response.citations:
                title = f" — {c.title}" if c.title else ""
                url = f" ({c.url})" if c.url else ""
                print(f"  [{c.marker}] {c.source}{title}{url}")
        if response.disclaimer:
            print(f"\n{response.disclaimer}")
        print("-" * 70)


if __name__ == "__main__":
    main()
