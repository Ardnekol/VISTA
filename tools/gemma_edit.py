#!/usr/bin/env python3
"""Call local Ollama/Gemma and replace a file with the model's edited output.

Usage:
  python3 tools/gemma_edit.py --file path/to/file --instruction "Make X changes"

Defaults to http://localhost:11434; change with --host if needed.
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

try:
    import requests
except Exception:
    print("Please install requests: pip install requests")
    raise


MARK_START = "<<<START>>>"
MARK_END = "<<<END>>>"


def call_gemma(host: str, model: str, prompt: str, timeout: int = 120) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def extract_between(text: str) -> str:
    # Prefer explicit markers, otherwise return whole response
    m = re.search(re.escape(MARK_START) + r"(.*)" + re.escape(MARK_END), text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="File to edit")
    p.add_argument("--instruction", required=True, help="Instruction for model")
    p.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    p.add_argument("--model", default="gemma4:31b", help="Model name to call")
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    filepath = os.path.abspath(args.file)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(2)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    prompt = (
        "You are an expert code editor. Apply the user's instruction to the file.\n"
        "Return only the full updated file contents between the markers below, with no extra explanation.\n"
        f"{MARK_START}\n"
        f"FILEPATH: {filepath}\n"
        f"---FILE START---\n{content}\n---FILE END---\n"
        f"INSTRUCTION: {args.instruction}\n"
        f"{MARK_END}\n"
    )

    print("Calling gemma at", args.host, "(this may take a few seconds)")
    try:
        resp = call_gemma(args.host, args.model, prompt, timeout=args.timeout)
    except Exception as e:
        print("Error calling gemma:", e)
        sys.exit(1)

    new_content = extract_between(resp)
    if not new_content:
        print("Model returned empty response. Aborting.")
        sys.exit(1)

    # Backup original
    bak_path = filepath + ".bak"
    shutil.copy2(filepath, bak_path)
    print(f"Backup written to: {bak_path}")

    # Write new content safely
    fd, tmp_path = tempfile.mkstemp(prefix="gemma_edit_", suffix=".tmp")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    shutil.move(tmp_path, filepath)
    print(f"File updated: {filepath}")


if __name__ == "__main__":
    main()
