"""Re-judge saved IFBench outputs with the official checkers: as returned, and without the leading newlines after </think>."""
import ast, json, sys
from evalscope.benchmarks.ifbench import instructions_registry as reg


def passes(iid, kw, prompt, resp):
    ins = reg.INSTRUCTION_DICT[iid](iid)
    ins.build_description(**{k: v for k, v in kw.items() if v is not None})
    a = ins.get_instruction_args()
    if a and "prompt" in a:
        ins.build_description(prompt=prompt)
    return bool(resp.strip() and ins.check_following(resp))


rows = [json.loads(l) for l in open(sys.argv[1])]
a = b = 0; gain = []; lose = []; head = {}
for r in rows:
    s = r["sample_score"]; m = s["sample_metadata"]; pred = s["score"]["prediction"] or ""
    ids = m["instruction_id_list"]; ids = ast.literal_eval(ids) if isinstance(ids, str) else ids
    kws = m["kwargs"]; kws = ast.literal_eval(kws) if isinstance(kws, str) else kws
    x = all(passes(i, k, m["prompt"], pred) for i, k in zip(ids, kws))
    y = all(passes(i, k, m["prompt"], pred.lstrip("\n")) for i, k in zip(ids, kws))
    if bool(x) != (s["score"]["value"]["prompt_level_strict"] == 1):
        print("MISMATCH with the stored score", r["index"], ids)
    a += x; b += y
    head[repr(pred[:2])] = head.get(repr(pred[:2]), 0) + 1
    if y and not x: gain.append((r["index"], ids))
    if x and not y: lose.append((r["index"], ids))
print("as returned", a, "/", len(rows)); print("leading newlines removed", b, "/", len(rows))
print("gain", gain); print("lose", lose); print("first two chars", sorted(head.items(), key=lambda t: -t[1])[:5])
