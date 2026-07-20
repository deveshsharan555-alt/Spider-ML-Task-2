"""Terminal chat for the Healthcare Information Assistant."""

import time
from rag_core import Assistant


def main():
    print("Building knowledge base...")
    bot = Assistant()
    t0 = time.time()
    bot.build()
    print(f"Loaded {bot.num_documents} documents -> {bot.num_chunks} chunks in {time.time() - t0:.2f}s\n")
    print("Type 'exit' to quit. (General information only, not medical advice.)\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = bot.ask(query)
        print(f"\nAssistant: {result['answer']}\n")

        if result["confidence"]:
            c = result["confidence"]
            print(f"Confidence: {c['label']} ({c['score']}) — {c['rationale']}")

        if result["citations"]:
            print("Sources:")
            for c in result["citations"]:
                title = f" — {c['title']}" if c["title"] else ""
                print(f"  [{c['marker']}] {c['source']}{title}")

        print("-" * 60)


if __name__ == "__main__":
    main()
