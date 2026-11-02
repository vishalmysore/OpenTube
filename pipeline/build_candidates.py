"""Join stage-0 output into one candidates file: every comment that links a YouTube video, with its thread (story)
title, plus per-video counts of how many distinct people linked it.

Usage: python3 -I build_candidates.py --work data/work --out data/candidates.jsonl
"""
import argparse, glob, json, os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--out", default="data/candidates.jsonl")
    a = ap.parse_args()

    par = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(a.work, "par_*.npy")))])
    order = np.argsort(par[:, 0])
    cid, cpar = par[order, 0], par[order, 1]
    print("comments in archive:", len(cid))

    yt = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(a.work, "yt_*.parquet")))], promote_options="permissive")
    rows = yt.to_pylist()
    print("comments linking YouTube:", len(rows))

    # Walk each comment up to its story: while the current id is a comment, step to its parent.
    cur = np.array([r["parent"] for r in rows], dtype=np.uint32)
    depth = np.zeros(len(cur), dtype=np.int32)
    for _ in range(200):
        pos = np.searchsorted(cid, cur)
        pos = np.minimum(pos, len(cid) - 1)
        is_comment = cid[pos] == cur
        if not is_comment.any():
            break
        cur = np.where(is_comment, cpar[pos], cur)
        depth += is_comment

    del par, order, cid, cpar  # free ~0.7 GB before loading stories
    sch = pa.schema([("id", pa.uint32()), ("title", pa.string()), ("url", pa.string()), ("score", pa.int32())])
    st = pa.concat_tables([pq.read_table(f, columns=sch.names).cast(sch) for f in sorted(glob.glob(os.path.join(a.work, "st_*.parquet")))])
    print("stories in archive:", st.num_rows)
    need = pa.array(np.unique(cur), type=pa.uint32())
    st = st.filter(pc.is_in(st["id"], value_set=need))
    stories = {r["id"]: r for r in st.to_pylist()}

    linkers = {}
    for r in rows:
        for v in r["video_ids"]:
            linkers.setdefault(v, set()).add(r["by"])

    n_found = 0
    with open(a.out, "w") as f:
        for r, sid, d in zip(rows, cur.tolist(), depth.tolist()):
            s = stories.get(sid)
            n_found += s is not None
            f.write(json.dumps({
                "id": r["id"], "by": r["by"], "time": r["time"].strftime("%Y-%m-%d"), "parent": r["parent"],
                "story_id": sid, "story_title": (s or {}).get("title", ""), "story_url": (s or {}).get("url", ""),
                "story_score": (s or {}).get("score", 0), "depth": d, "video_ids": r["video_ids"],
                "max_linkers": max(len(linkers[v]) for v in r["video_ids"]), "text": r["text"],
            }) + "\n")
    print("thread title found for", n_found, "of", len(rows))
    counts = np.array([len(s) for s in linkers.values()])
    print("distinct videos:", len(counts), {k: int((counts >= k).sum()) for k in (1, 2, 3, 5, 10)})


if __name__ == "__main__":
    main()
