"""Ask the incident knowledge base from the terminal.

Run:  python -m rag.cli
(Requires the index to be built first: python -m rag.ingest)
"""
from __future__ import annotations

from dotenv import load_dotenv

from .pipeline import answer


def main() -> None:
    load_dotenv()
    print("Incident-RAG — ask about build/test/deploy failures. Ctrl-C to quit.")
    print("Example: 'How do we fix OrderServiceIT timeouts against hana-test?'\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        result = answer(q)
        print("\nassistant >", result["answer"])
        print("citations:", ", ".join(result["citations"]), "\n")


if __name__ == "__main__":
    main()
