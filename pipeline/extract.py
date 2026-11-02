"""Stage 0 of OpenTube Graph: stream the whole Hacker News archive (open-index/hacker-news on Hugging Face,
one Parquet file per month), keep only what later stages need, and delete each month after reading it.

Outputs (in data/work/):
  yt_comments.parquet  comments whose text links to a YouTube video (id, by, time, parent, text, video_ids)
  parents.npy          uint32 pairs (id, parent) for every comment, used to find each comment's story
  stories.parquet      every story: id, title, url, score, time, by
  extract_log.json     per-month row counts, bytes downloaded and seconds spent (stage-0 cost)

Usage: python3 -I extract.py --out data/work [--from 2006-10] [--to 2026-08]
"""
import argparse, json, os, re, sys, time, urllib.request
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/open-index/hacker-news/resolve/main/"
STATS = BASE + "stats.csv"
# YouTube video ids are 11 chars of [A-Za-z0-9_-]. HN stores comment HTML with escaped entities.
YT_RE = re.compile(r"(?:youtube\.com/(?:watch\?(?:[^\"'<>\s]*?&(?:amp;)?)?v=|embed/|shorts/|live/|v/)|youtu\.be/)([A-Za-z0-9_-]{11})")
YT_HINT = r"youtu"


def months(lo, hi):
    with urllib.request.urlopen(STATS, timeout=60) as r:
        rows = r.read().decode().strip().splitlines()[1:]
    out = []
    for line in rows:
        y, m, _lo, _hi, count, *_rest, size, _c = line.split(",")
        key = f"{int(y):04d}-{int(m):02d}"
        if lo <= key <= hi and int(count) > 0:
            out.append((key, int(size)))
    return out


def fetch(url, dest, tries=4):
    for i in range(tries):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # network blips: retry with backoff
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/work")
    ap.add_argument("--from", dest="lo", default="2006-01")
    ap.add_argument("--to", dest="hi", default="2099-12")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_month.parquet")
    log_path = os.path.join(a.out, "extract_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    cols = ["id", "type", "by", "time", "text", "parent", "title", "url", "score", "dead", "deleted"]

    for key, size in months(a.lo, a.hi):
        if key in log:
            continue
        y, m = key.split("-")
        t0 = time.time()
        fetch(f"{BASE}data/{y}/{y}-{int(m):02d}.parquet", tmp)
        t1 = time.time()
        t = pq.read_table(tmp, columns=cols)
        live = pc.and_(pc.equal(t["deleted"], 0), pc.equal(t["dead"], 0))
        comments = t.filter(pc.and_(live, pc.equal(t["type"], 2)))
        stories = t.filter(pc.equal(t["type"], 1)).select(["id", "title", "url", "score", "time", "by"])
        par = np.stack([comments["id"].to_numpy(), comments["parent"].to_numpy()], axis=1).astype(np.uint32)
        cand = comments.filter(pc.match_substring(comments["text"], YT_HINT))
        rows = []
        for r in cand.select(["id", "by", "time", "parent", "text"]).to_pylist():
            # Some months store link targets with escaped slashes (http:&#x2F;&#x2F;...), so unescape before matching.
            plain = r["text"].replace("&#x2F;", "/").replace("&#47;", "/").replace("&amp;", "&")
            ids = list(dict.fromkeys(YT_RE.findall(plain)))
            if ids:
                r["video_ids"] = ids
                rows.append(r)
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), os.path.join(a.out, f"yt_{key}.parquet"))
        pq.write_table(stories, os.path.join(a.out, f"st_{key}.parquet"))
        np.save(os.path.join(a.out, f"par_{key}.npy"), par)
        os.remove(tmp)
        log[key] = {"items": t.num_rows, "comments": comments.num_rows, "yt_comments": len(rows),
                    "bytes": size, "download_s": round(t1 - t0, 1), "process_s": round(time.time() - t1, 1)}
        json.dump(log, open(log_path, "w"), indent=0)
        print(key, log[key], flush=True)


if __name__ == "__main__":
    main()
