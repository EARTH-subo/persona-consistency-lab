# -*- coding: utf-8 -*-
"""姬子官方语料 · 混合检索器（FTS5 全文 + BGE 向量，RRF 融合）。

被 jizi_agent 以子进程方式调用（用 myenv 的 python 跑），故只依赖 myenv 已装库。
用法：
  人类可读：& myenv\\python.exe retrieve.py "隆介是谁"
  JSON 输出：& myenv\\python.exe retrieve.py --json "隆介是谁"
"""
import json
import sqlite3
import sys
from pathlib import Path

import jieba
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
DB = HERE / "himeko_fts.db"
VEC = HERE / "embeddings.npy"
META = HERE / "embeddings_meta.jsonl"
CORPUS = HERE / "corpus.jsonl"
MODEL = HERE / "models" / "bge-small-zh-v1.5"
CHUNK, OVERLAP, MAXLEN = 300, 60, 256

USERWORDS = ["姬子", "素子", "无量塔", "星穹列车", "二相乐园", "拓星者", "匹诺康尼",
             "翁法罗斯", "罗浮", "雅利洛", "告死魔", "归寂", "幻月游戏", "风化诅咒",
             "无名客", "开拓", "知更鸟", "黄泉", "花火", "桑博", "真珠", "符玄",
             "驭空", "青雀", "布洛妮娅", "希露瓦", "银狼", "卡芙卡", "流萤", "绘世", "隆介"]
for _w in USERWORDS:
    jieba.add_word(_w)


class _BGE:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        self.tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        self.tok.enable_truncation(max_length=MAXLEN)
        self.tok.enable_padding(length=None)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(MODEL / "onnx" / "model.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts):
        encs = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0]
        v = out[:, 0]
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _split(t):
    t = t.strip()
    if len(t) <= CHUNK:
        return [t] if t else []
    out, i = [], 0
    while i < len(t):
        out.append(t[i:i + CHUNK])
        i += CHUNK - OVERLAP
    return out


_DOCS = None


def _docs():
    global _DOCS
    if _DOCS is None:
        _DOCS = {}
        for line in CORPUS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for j, ch in enumerate(_split(r["text"])):
                _DOCS[f"{r['title']}#{j}"] = ch
    return _DOCS


def fts_search(q, k):
    toks = [t for t in jieba.cut(q) if t.strip()]
    if not toks:
        return []
    match = " OR ".join(f'"{t}"' for t in toks)
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT title,region,version,url,raw FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
        (match, k)).fetchall()
    con.close()
    return [{"title": r[0], "region": r[1], "version": r[2], "url": r[3],
             "text": r[4], "src": "fts"} for r in rows]


def vec_search(q, k):
    arr = np.load(VEC)
    metas = [json.loads(l) for l in META.read_text(encoding="utf-8").splitlines() if l.strip()]
    qv = _BGE.get().encode([q])[0]
    sim = arr @ qv
    idx = np.argsort(-sim)[:k]
    docs = _docs()
    return [{"title": metas[i]["title"], "region": metas[i]["region"],
             "version": metas[i]["version"], "url": metas[i]["url"],
             "text": docs.get(metas[i]["cid"], ""), "src": "vec",
             "score": float(sim[i])} for i in idx]


def retrieve(q, k=5):
    """FTS + 向量各取 2k，用 RRF 融合，返回 top-k。"""
    cand = fts_search(q, k * 2) + vec_search(q, k * 2)
    fused, order = {}, []
    for rank, it in enumerate(cand):
        key = (it["title"], it["text"][:60])
        if key not in fused:
            fused[key] = {**it, "rrf": 0.0}
            order.append(key)
        fused[key]["rrf"] += 1.0 / (60 + rank)
    merged = sorted((fused[key] for key in order), key=lambda x: -x["rrf"])
    return merged[:k]


def main():
    args = sys.argv[1:]
    as_json = "--json" in args
    if as_json:
        args.remove("--json")
    q = " ".join(args)
    hits = retrieve(q)
    if as_json:
        print(json.dumps(hits, ensure_ascii=False))
        return
    for h in hits:
        print(f"\n[{h['rrf']:.4f}] 【{h['title']}】{h['region']} {h['version']} ({h['src']})\n"
              f"  …{h['text'][:160]}…\n  {h['url']}")


if __name__ == "__main__":
    main()
