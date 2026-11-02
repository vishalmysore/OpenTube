"""Repeated 2-fold cross-validation of the threshold choices on the labelled sample, so the published accuracy is not
only the in-sample (optimistic) number. Writes data/cv.json.  python3 -I pipeline/cv.py"""
import json, random
lab = {o["id"]: o["label"] == "recommends" for o in json.load(open("data/labels.json"))}
R = [json.loads(l) for l in open("data/out/sample_both.jsonl")]
for r in R: r["y"] = lab[r["id"]]
G = [i / 100 for i in range(30, 80)]
T2 = [i / 100 for i in range(5, 80)]
acc = lambda S, f: sum(f(r) == r["y"] for r in S) / len(S)
def casc(r, lo, hi, t): return True if r["laya_p"] >= hi else False if r["laya_p"] <= lo else r["llm_p"] >= t
def fit_single(S, k): return max(G + T2, key=lambda t: acc(S, lambda r: r[k] >= t))
def fit_casc(S, maxllm=0.35):
    best = None
    for lo in G:
        for hi in G:
            if hi <= lo: continue
            frac = sum(lo < r["laya_p"] < hi for r in S) / len(S)
            if frac > maxllm: continue
            for t in T2:
                a = acc(S, lambda r: casc(r, lo, hi, t))
                if best is None or a > best[0]: best = (a, lo, hi, t)
    return best
random.seed(2026)
res = {"laya": [], "llm": [], "cascade": [], "to_llm": []}
for rep in range(10):
    S = R[:]; random.shuffle(S); A, B = S[:150], S[150:]
    for tr, te in ((A, B), (B, A)):
        t = fit_single(tr, "laya_p"); res["laya"].append(acc(te, lambda r: r["laya_p"] >= t))
        t = fit_single(tr, "llm_p"); res["llm"].append(acc(te, lambda r: r["llm_p"] >= t))
        _, lo, hi, t = fit_casc(tr); res["cascade"].append(acc(te, lambda r: casc(r, lo, hi, t)))
        res["to_llm"].append(sum(lo < r["laya_p"] < hi for r in te) / len(te))
out = {k: round(sum(v) / len(v), 4) for k, v in res.items()}
out["method"] = "10 x repeated 2-fold CV: thresholds fit on 150 comments, scored on the other 150"
json.dump(out, open("data/cv.json", "w"), indent=1)
print(out)
