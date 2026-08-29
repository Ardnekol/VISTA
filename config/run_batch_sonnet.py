#!/usr/bin/env python3
"""
VISDA decision analysis — Anthropic Claude Sonnet 5 via the Message Batches API.
================================================================================
Frontier Anthropic closed-model comparison point. Built to be CREDIT-SAFE:
you can run it on a small balance, let it exhaust the credit, top up, and
re-run/--resume to finish — without losing or re-paying for completed work.

Reuses ALL shared logic from run_batch_haiku.py (prompt, dot-product baseline,
dataset, assemble_rows, FIELDNAMES) so it's apples-to-apples with every other
model: same 8,550 decisions -> master_llm_decisions_sonnet.csv.

Sonnet-5 specifics (differ from the Haiku script):
  - NO temperature (Sonnet 5 rejects non-default sampling params -> 400).
  - thinking = {"type": "disabled"} — Sonnet 5 has adaptive thinking ON by
    default; leaving it on would emit thinking tokens billed as output and
    inflate cost. Disabling keeps it non-reasoning + cheap, matching the others.

CREDIT-SAFE RESUME MODEL:
  - Only SUCCESSFUL decisions are cached (results_cache_sonnet.jsonl).
  - Each run reprocesses only cids NOT already cached, in small chunks, so a
    partially-completed job resumes exactly where it stopped, no double charge.
  - In-flight batch ids are persisted (pending_batches_sonnet.json) so a crash
    while waiting re-attaches instead of resubmitting.
  - If credit runs out: the run stops (submit 400s). Top up, run again — it
    skips everything already cached and continues.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 run_batch_sonnet.py            # process all uncached work, write CSV
  python3 run_batch_sonnet.py            # run again after a top-up -> continues
  python3 run_batch_sonnet.py --dry-run  # show remaining/chunk plan, no submit
"""

import json
import csv
import time
import argparse

import anthropic
import run_batch_haiku as R   # shared dataset + prompt + dp + assemble_rows + FIELDNAMES

MODEL       = "claude-sonnet-5"
MAX_TOKENS  = 512
SUFFIX      = "sonnet"
CHUNK_SIZE  = 1000            # small commits -> fine-grained credit recovery

OUT_DIR      = R.OUT_DIR
CACHE_FILE   = OUT_DIR / f"results_cache_{SUFFIX}.jsonl"    # success-only: {"cid","res"}
PENDING_FILE = OUT_DIR / f"pending_batches_{SUFFIX}.json"   # [batch_id, ...] in-flight
MASTER_OUT   = OUT_DIR / f"master_llm_decisions_{SUFFIX}.csv"

STRUCT = {"format": {"type": "json_schema", "schema": R.DECISION_SCHEMA}}


# ─── request bodies ───────────────────────────────────────────────────────────

def unit_to_request(u):
    sc = u["scenario"]
    mod_text = u["modifier"]["modifier_text"] if u["modifier"] else None
    prompt = R.build_prompt(u["desc"], sc["description"], sc["A0"], sc["A1"], mod_text)
    return {
        "custom_id": u["cid"],
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "thinking": {"type": "disabled"},   # no thinking tokens; matches other models
            "messages": [{"role": "user", "content": prompt}],
            "output_config": STRUCT,
        },
    }


def parse_content(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
        text = text.strip()
    return json.loads(text)


# ─── cache + pending state ────────────────────────────────────────────────────

def load_cache():
    cache = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text().splitlines():
            if line.strip():
                o = json.loads(line); cache[o["cid"]] = o["res"]
    return cache

def append_cache(cid, res):
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"cid": cid, "res": res}) + "\n")

def load_pending():
    return json.loads(PENDING_FILE.read_text()) if PENDING_FILE.exists() else []

def save_pending(ids):
    PENDING_FILE.write_text(json.dumps(ids))


# ─── batch lifecycle ──────────────────────────────────────────────────────────

def wait_ended(client, batch_id, poll):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"    status={b.processing_status}  proc={c.processing} "
              f"ok={c.succeeded} err={c.errored} exp={c.expired}", flush=True)
        if b.processing_status == "ended":
            return b
        time.sleep(poll)


def collect(client, batch_id, cache):
    """Cache ONLY successful decisions. Errored/expired cids stay uncached -> retried."""
    added = 0
    for res in client.messages.batches.results(batch_id):
        cid = res.custom_id
        if cid in cache:
            continue
        if res.result.type == "succeeded":
            text = next((b.text for b in res.result.message.content if b.type == "text"), "")
            try:
                out = parse_content(text)
            except Exception as e:
                out = {"decision": "ERROR", "confidence": "low",
                       "driving_values": [], "reasoning": f"parse failed: {e}"}
            # a clean parse is a real success; a parse-failure is rare & cacheable
            cache[cid] = out; append_cache(cid, out); added += 1
        # errored / expired / canceled -> DO NOT cache; leave for a later resume
    return added


def submit(client, units_chunk):
    reqs = [unit_to_request(u) for u in units_chunk]
    batch = client.messages.batches.create(requests=reqs)
    return batch.id


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="(default behavior — kept for clarity)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    profiles     = R.load_profiles()
    scenarios    = json.loads(R.SCENARIOS_FILE.read_text())
    modifiers_db = json.loads(R.MODIFIERS_FILE.read_text())
    modifier_lookup = {m["scenario_id"]: m.get("modifiers", []) for m in modifiers_db}
    units = R.build_units(profiles, scenarios, modifier_lookup)

    cache   = load_cache()
    pending = load_pending()
    remaining = [u for u in units if u["cid"] not in cache]
    print(f"Requests total: {len(units)}  cached: {len(cache)}  remaining: {len(remaining)}  "
          f"pending batches: {len(pending)}  (model: {MODEL})")

    if args.dry_run:
        for i in range(0, len(remaining), CHUNK_SIZE):
            ch = remaining[i:i + CHUNK_SIZE]
            print(f"  chunk: {len(ch)} requests  ({ch[0]['cid']} .. {ch[-1]['cid']})")
        return

    client = anthropic.Anthropic()

    try:
        # 1) drain any in-flight batches from a previous interrupted run
        for bid in list(pending):
            print(f"[pending] re-attaching batch {bid}")
            wait_ended(client, bid, args.poll)
            added = collect(client, bid, cache)
            pending.remove(bid); save_pending(pending)
            print(f"[pending] collected (+{added}, {len(cache)} cached)")

        # 2) process remaining work in chunks
        remaining = [u for u in units if u["cid"] not in cache]
        for i in range(0, len(remaining), CHUNK_SIZE):
            chunk = [u for u in remaining[i:i + CHUNK_SIZE] if u["cid"] not in cache]
            if not chunk:
                continue
            print(f"[submit] {len(chunk)} requests...")
            bid = submit(client, chunk)                 # <- 400s here if credit is out
            pending.append(bid); save_pending(pending)
            print(f"[submit] batch {bid}")
            wait_ended(client, bid, args.poll)
            added = collect(client, bid, cache)
            pending.remove(bid); save_pending(pending)
            print(f"[submit] collected (+{added}, {len(cache)}/{len(units)} cached)")

    except anthropic.APIStatusError as e:
        print(f"\n[!] API error: {e}")
        print("If this is a credit/billing error: add credit on the Anthropic console, "
              "then re-run `python3 run_batch_sonnet.py` — it resumes from the cache, "
              "no double charge.")
        raise SystemExit(1)

    # 3) assemble master from cache (uncached cids -> ERROR rows, if any)
    left = [u for u in units if u["cid"] not in cache]
    if left:
        print(f"\n[!] {len(left)} requests still uncached (errored/expired). "
              f"Re-run to retry them before treating the CSV as final.")

    rows, flip_llm, flip_dp, n_mod = R.assemble_rows(
        profiles, scenarios, modifier_lookup, cache)
    with open(MASTER_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=R.FIELDNAMES); w.writeheader(); w.writerows(rows)

    n_err = sum(1 for r in rows if r["llm_decision"] == "ERROR")
    print(f"\n{'='*55}")
    print(f"  Rows written:      {len(rows)}  -> {MASTER_OUT}")
    print(f"  ERROR/uncached:    {n_err}")
    print(f"  Modifier trials:   {n_mod}")
    print(f"  LLM flips:         {flip_llm} / {n_mod}  ({100*flip_llm/n_mod:.2f}%)")
    print(f"  Dot-product flips: {flip_dp} / {n_mod}  ({100*flip_dp/n_mod:.2f}%)")
    print(f"{'='*55}\n")
    print(f'Next: add to step3/step6 -> "Sonnet 5": OUT_DIR / "master_llm_decisions_{SUFFIX}.csv"')


if __name__ == "__main__":
    main()
