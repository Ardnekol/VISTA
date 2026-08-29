#!/usr/bin/env python3
"""
VISDA decision analysis — OpenAI GPT-5-mini via the Batch API (chunked).
========================================================================
Current-generation (GPT-5 family) closed-model comparison point.

Reuses ALL shared logic from run_batch_haiku.py (prompt, dot-product baseline,
dataset, assemble_rows, FIELDNAMES) so it stays apples-to-apples with the other
models: same 95 profiles x 10 scenarios x (1 baseline + 8 modifiers) = 8,550
decisions -> master_llm_decisions_gpt5mini.csv.

GPT-5 specifics (differ from the GPT-4.1-mini script):
  - reasoning model: reasoning_effort="minimal" (binary A0/A1 + one-line reason
    needs no deep reasoning; keeps reasoning-token cost/latency low).
  - NO temperature (GPT-5 reasoning models reject a custom temperature) -> not
    strictly greedy; note this reproducibility caveat in the paper (same class of
    note as any reasoning model).
  - max_completion_tokens (NOT max_tokens) — this budget INCLUDES hidden reasoning
    tokens, so it's set with headroom to avoid truncated (empty) answers.
  - smaller chunks: OpenAI's ~2M enqueued-token cap counts input + the reserved
    max_completion_tokens, and our reservation is larger here, so CHUNK_SIZE is
    smaller than the GPT-4.1 run.

Resumable: per-chunk batch ids in a state file + a decision cache, so --resume
skips finished work and never double-charges.

Usage:
  export OPENAI_API_KEY=sk-...
  python3 run_batch_gpt5mini.py            # run all chunks, then write CSV
  python3 run_batch_gpt5mini.py --resume   # continue after an interruption
  python3 run_batch_gpt5mini.py --dry-run  # show chunk plan, don't submit
"""

import json
import csv
import time
import argparse

from openai import OpenAI
import run_batch_haiku as R   # shared dataset + prompt + dp + assemble_rows + FIELDNAMES

MODEL            = "gpt-5-mini"      # change here if you want a different GPT-5 variant
REASONING_EFFORT = "minimal"
MAX_COMPLETION   = 1500              # includes reasoning tokens; headroom vs truncation
SUFFIX           = "gpt5mini"
CHUNK_SIZE       = 600               # ~600 * (~950 in + 1500 reserved) ~= 1.47M < 2M cap

OUT_DIR    = R.OUT_DIR
STATE_FILE = OUT_DIR / f"batch_state_{SUFFIX}.json"     # {chunk_index: batch_id}
CACHE_FILE = OUT_DIR / f"results_cache_{SUFFIX}.jsonl"  # one {cid: ...} per line
MASTER_OUT = OUT_DIR / f"master_llm_decisions_{SUFFIX}.csv"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "decision", "strict": True, "schema": R.DECISION_SCHEMA},
}
TERMINAL = {"completed", "failed", "expired", "cancelled"}


# ─── request + result plumbing ────────────────────────────────────────────────

def unit_to_line(u):
    sc = u["scenario"]
    mod_text = u["modifier"]["modifier_text"] if u["modifier"] else None
    prompt = R.build_prompt(u["desc"], sc["description"], sc["A0"], sc["A1"], mod_text)
    return {
        "custom_id": u["cid"], "method": "POST", "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": REASONING_EFFORT,
            "max_completion_tokens": MAX_COMPLETION,
            "response_format": RESPONSE_FORMAT,
        },
    }


def parse_content(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
        text = text.strip()
    return json.loads(text)


def load_cache():
    cache = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text().splitlines():
            if line.strip():
                obj = json.loads(line); cache[obj["cid"]] = obj["res"]
    return cache

def append_cache(cid, res):
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"cid": cid, "res": res}) + "\n")

def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


def wait_for_batch(client, batch_id, poll):
    while True:
        b = client.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"    status={b.status}  total={c.total} completed={c.completed} failed={c.failed}",
              flush=True)
        if b.status in TERMINAL:
            return b
        time.sleep(poll)


def collect_batch(client, batch, cache):
    added = 0
    if batch.output_file_id:
        for line in client.files.content(batch.output_file_id).text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line); cid = obj["custom_id"]
            try:
                if obj.get("error"):
                    raise ValueError(obj["error"])
                choice = obj["response"]["body"]["choices"][0]
                if choice["message"].get("content") is None:
                    raise ValueError(f"no content (finish={choice.get('finish_reason')})")
                res = parse_content(choice["message"]["content"])
            except Exception as e:
                res = {"decision": "ERROR", "confidence": "low",
                       "driving_values": [], "reasoning": f"parse/response error: {e}"}
            if cid not in cache:
                cache[cid] = res; append_cache(cid, res); added += 1
    if batch.error_file_id:
        for line in client.files.content(batch.error_file_id).text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line); cid = obj.get("custom_id")
            if cid and cid not in cache:
                res = {"decision": "ERROR", "confidence": "low",
                       "driving_values": [], "reasoning": f"batch error: {obj.get('error')}"}
                cache[cid] = res; append_cache(cid, res); added += 1
    return added


def submit_chunk(client, chunk, idx):
    path = OUT_DIR / f"batch_input_{SUFFIX}_{idx:02d}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for u in chunk:
            f.write(json.dumps(unit_to_line(u)) + "\n")
    up = client.files.create(file=open(path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
    return batch.id


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    profiles     = R.load_profiles()
    scenarios    = json.loads(R.SCENARIOS_FILE.read_text())
    modifiers_db = json.loads(R.MODIFIERS_FILE.read_text())
    modifier_lookup = {m["scenario_id"]: m.get("modifiers", []) for m in modifiers_db}

    units  = R.build_units(profiles, scenarios, modifier_lookup)
    chunks = [units[i:i + CHUNK_SIZE] for i in range(0, len(units), CHUNK_SIZE)]
    print(f"Profiles: {len(profiles)}  Requests: {len(units)}  "
          f"Chunks: {len(chunks)} x <= {CHUNK_SIZE}  (model: {MODEL}, reasoning={REASONING_EFFORT})")

    if args.dry_run:
        for i, ch in enumerate(chunks):
            print(f"  chunk {i:02d}: {len(ch)} requests  ({ch[0]['cid']} .. {ch[-1]['cid']})")
        return

    client = OpenAI()
    cache  = load_cache()
    state  = load_state()
    print(f"Cached decisions: {len(cache)}  |  Known chunk batches: {len(state)}")

    for i, chunk in enumerate(chunks):
        cids = [u["cid"] for u in chunk]
        if all(c in cache for c in cids):
            print(f"[chunk {i:02d}] all {len(cids)} cached — skip")
            continue

        bid = state.get(str(i))
        if bid:
            b = client.batches.retrieve(bid)
            if b.status not in TERMINAL:
                print(f"[chunk {i:02d}] resuming batch {bid}")
                b = wait_for_batch(client, bid, args.poll)
        else:
            print(f"[chunk {i:02d}] submitting {len(chunk)} requests...")
            bid = submit_chunk(client, chunk, i)
            state[str(i)] = bid; save_state(state)
            print(f"[chunk {i:02d}] batch {bid}")
            b = wait_for_batch(client, bid, args.poll)

        if b.status == "failed":
            print(f"[chunk {i:02d}] FAILED: {b.errors}")
            raise SystemExit("A chunk failed — inspect errors above, fix, and --resume.")
        added = collect_batch(client, b, cache)
        print(f"[chunk {i:02d}] collected (+{added} new, {len(cache)} total)")

    # ── assemble master ──
    rows, flip_llm, flip_dp, n_mod = R.assemble_rows(
        profiles, scenarios, modifier_lookup, cache)
    with open(MASTER_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=R.FIELDNAMES)
        w.writeheader(); w.writerows(rows)

    n_err = sum(1 for r in rows if r["llm_decision"] == "ERROR")
    print(f"\n{'='*55}")
    print(f"  Rows written:      {len(rows)}  -> {MASTER_OUT}")
    print(f"  ERROR rows:        {n_err}")
    print(f"  Modifier trials:   {n_mod}")
    print(f"  LLM flips:         {flip_llm} / {n_mod}  ({100*flip_llm/n_mod:.2f}%)")
    print(f"  Dot-product flips: {flip_dp} / {n_mod}  ({100*flip_dp/n_mod:.2f}%)")
    print(f"{'='*55}\n")
    if n_err:
        print(f"[!] {n_err} ERROR rows — likely reasoning-token truncation. "
              f"Bump MAX_COMPLETION and rerun (cached rows are skipped), or retry those rows.")
    print(f'Next: add to step3/step6 -> "GPT-5-mini": OUT_DIR / "master_llm_decisions_{SUFFIX}.csv"')


if __name__ == "__main__":
    main()
