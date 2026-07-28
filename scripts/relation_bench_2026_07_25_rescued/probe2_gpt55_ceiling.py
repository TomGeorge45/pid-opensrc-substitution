"""Probe 2 ceiling test: GPT-5.5-low on the same 19 hand-verified pairs.

Runs BOTH prompt variants Qwen saw (short forced YES/NO, and CoT v2 with the
explicit endpoint check) so the comparison is apples-to-apples in each direction.
World A (capability exists at scale) vs World B (task malformed on these crops).
"""
import base64
import json
import re
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from openai import OpenAI

env = {}
for line in Path("/Users/tomgeorge/pid-ml/.env").read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")

client = OpenAI(api_key=env["OPENAI_API_KEY"])

BUNDLE_DIR = Path(__file__).parent / "probe_bundle"
if not BUNDLE_DIR.exists():
    zp = hf_hub_download(repo_id="timthy45/pnid-extraction-datasets",
                         filename="benchmarks/probe_bundle_2026-07-24.zip",
                         repo_type="dataset", token=env["HF_TOKEN"])
    BUNDLE_DIR.mkdir(parents=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(BUNDLE_DIR)

answer_key = json.loads((BUNDLE_DIR / "answer_key.json").read_text())

PROBE2_PROMPT = (
    "This image shows a crop of a P&ID drawing. Two symbols are highlighted with colored "
    "boxes: a RED box around one entity, and a BLUE box around a second entity. Is there a "
    "physical pipe or line directly connecting the RED-boxed entity to the BLUE-boxed "
    "entity? Answer with EXACTLY one word: YES or NO."
)
PROBE2_COT_PROMPT_V2 = (
    "This image shows a crop of a P&ID drawing with a RED box around one entity and a BLUE "
    "box around a second entity. Trace any pipe/line starting from the RED box, step by "
    "step, briefly. Then explicitly state: does that traced line's endpoint match the "
    "BLUE-boxed entity specifically? Answer only after checking that. End with a new line: "
    "'ANSWER: YES' or 'ANSWER: NO'."
)


def gpt(prompt, img_path, max_tokens=2000):
    b64 = base64.standard_b64encode(img_path.read_bytes()).decode()
    resp = client.chat.completions.create(
        model="gpt-5.5", reasoning_effort="low",
        max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}])
    return (resp.choices[0].message.content or "").strip()


def parse_short(raw):
    t = raw.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    return None


def parse_cot(raw):
    m = re.search(r"ANSWER:\s*(YES|NO)", raw, re.IGNORECASE)
    return m.group(1).upper() == "YES" if m else None


def run(label, prompt, parser):
    rows = []
    for cand in answer_key:
        if cand["verdict"] == "SKIP":
            continue
        img = BUNDLE_DIR / cand["crop_file"]
        raw = gpt(prompt, img)
        pred = parser(raw)
        expected = cand["verdict"] == "TRUE"
        rows.append({"pair": img.name, "expected": cand["verdict"],
                     "pred": pred, "correct": pred == expected, "raw": raw})
        print(f"[{label}] {img.name}: expected={cand['verdict']:<5} pred={pred}  "
              f"correct={pred == expected}")
    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    n_unparse = sum(r["pred"] is None for r in rows)
    n_yes = sum(r["pred"] is True for r in rows)
    print(f"\n[{label}] accuracy: {n_correct}/{n} = {n_correct / n:.1%}  "
          f"(unparseable: {n_unparse}, said YES: {n_yes}/{n})\n")
    return rows


print("=== GPT-5.5-low ceiling test, 19 pairs ===\n")
short_rows = run("short YES/NO", PROBE2_PROMPT, parse_short)
cot_rows = run("CoT v2", PROBE2_COT_PROMPT_V2, parse_cot)

out = {"short": short_rows, "cot_v2": cot_rows}
out_path = Path(__file__).parent / "probe2_gpt55_ceiling_results.json"
out_path.write_text(json.dumps(out, indent=1))
print(f"raw results saved: {out_path}")

print("\n=== Reference: Qwen3-VL-8B on the same pairs (today) ===")
print("base, short prompt: 52.6% (constant NO)")
print("base, CoT v2:       9/17 = 52.9% on parseable subset")
print("v3-relation:        degenerate 'No.' repetition, both prompts")
