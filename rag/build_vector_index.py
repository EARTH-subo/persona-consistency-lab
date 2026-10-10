# -*- coding: utf-8 -*-
"""用 ONNX BGE-small-zh 向量化 corpus.jsonl，用 numpy 存/查（不依赖 chromadb）。

方案：流式分批编码 → 存 embeddings.npy + embeddings_meta.jsonl → 查询时暴力余弦（数据量小，够快）。
用法：
  建库：python build_vector_index.py
  查询：python build_vector_index.py "姬子和素子的关系"
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "bge-small-zh-v1.5"
VEC = HERE / "embeddings.npy"
META = HERE / "embeddings_meta.jsonl"
CHUNK, OVERLAP = 300, 60
MAXLEN, FLUSH = 256, 64


class BGE:
    def __init__(self, d: Path):
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=MAXLEN)
        self.tok.enable_padding(length=None)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(d / "onnx" / "model.onnx"), opts,
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


def split_text(t, size=CHUNK, overlap=OVERLAP):
    t = t.strip()
    if len(t) <= size:
        return [t] if t else []
    out, i = [], 0
    while i < len(t):
        out.append(t[i:i + size])
        i += size - overlap
    return out


def build():
    model = BGE(MODEL)
    vecs, metas, buf = [], [], []

    def flush():
        if not buf:
            return
        emb = model.encode([d for _, d in buf])
        for (m, _), v in zip(buf, emb):
            vecs.append(v)
            metas.append(m)
        print("encoded", len(vecs), flush=True)
        buf.clear()

    for line in (HERE / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for j, ch in enumerate(split_text(r["text"])):
            buf.append(({"title": r["title"], "region": r["region"], "version": r["version"],
                         "url": r["url"], "cid": f"{r['title']}#{j}"}, ch))
            if len(buf) >= FLUSH:
                flush()
    flush()
    arr = np.vstack(vecs).astype("float32")
    np.save(VEC, arr)
    with META.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"完成：{VEC} {arr.shape}｜{META}")


def query(q, k=5):
    arr = np.load(VEC)
    metas = [json.loads(l) for l in META.read_text(encoding="utf-8").splitlines() if l.strip()]
    qv = BGE(MODEL).encode([q])[0]
    sim = arr @ qv
    for i in np.argsort(-sim)[:k]:
        m = metas[i]
        doc = _doc_cache(m["cid"])
        print(f"\n[{sim[i]:.3f}] 【{m['title']}】{m['region']} {m['version']}\n  …{doc[:140]}…\n  {m['url']}")


_DOCS = None


def _doc_cache(cid):
    global _DOCS
    if _DOCS is None:
        _DOCS = {}
        for line in (HERE / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for j, ch in enumerate(split_text(r["text"])):
                _DOCS[f"{r['title']}#{j}"] = ch
    return _DOCS.get(cid, "")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
    else:
        build()
