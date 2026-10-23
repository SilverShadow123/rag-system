"""
Remove ragas' two langchain_community imports so ragas works WITHOUT
langchain-community installed.

Usage (inside your activated venv):
    pip install ragas
    pip uninstall -y langchain-community
    python patch_ragas.py

Safe to run more than once. Re-run it after upgrading or reinstalling ragas.
Only ragas/llms/base.py is touched; the removed names were only used in an
isinstance() list for VertexAI models, which you are not using.
"""
import importlib.util
import sys
from pathlib import Path

LINES_TO_REMOVE = [
    "from langchain_community.chat_models.vertexai import ChatVertexAI\n",
    "from langchain_community.llms import VertexAI\n",
]
LIST_ENTRIES = "    ChatVertexAI,\n    VertexAI,\n]"


def main() -> int:
    spec = importlib.util.find_spec("ragas")
    if spec is None or spec.origin is None:
        print("ragas is not installed in this environment.")
        return 1

    target = Path(spec.origin).parent / "llms" / "base.py"
    source = target.read_text(encoding="utf-8")
    patched = source

    for line in LINES_TO_REMOVE:
        patched = patched.replace(line, "")
    patched = patched.replace(LIST_ENTRIES, "]")

    if patched == source:
        print(f"Nothing to patch (already patched or different ragas version): {target}")
        return 0

    target.write_text(patched, encoding="utf-8")
    print(f"Patched: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())