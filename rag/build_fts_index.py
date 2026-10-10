# -*- coding: utf-8 -*-
"""把 corpus.jsonl 建成 FTS5 全文索引（jieba 分词 + unicode61），支持 1~2 字中文名检索。

输出：rag/himeko_fts.db
用法：
  建库：python build_fts_index.py
  查询：python build_fts_index.py "姬子和素子"
"""
import json
import sqlite3
import sys
from pathlib import Path

import jieba

HERE = Path(__file__).resolve().parent
DB = HERE / "himeko_fts.db"
CHUNK = 400
OVERLAP = 80

# 自定义词表：防止姬子/素子/专有名词被切碎
USERWORDS = ["姬子", "素子", "无量塔", "星穹列车", "二相乐园", "拓星者", "匹诺康尼",
             "翁法罗斯", "罗浮", "雅利洛", "告死魔", "归寂", "幻月游戏", "风化诅咒",
             "无名客", "开拓", "知更鸟", "黄泉", "花火", "桑博", "真珠", "符玄",
             "驭空", "青雀", "布洛妮娅", "希露瓦", "银狼", "卡芙卡", "流萤", "绘世", "隆介"]
for w in USERWORDS:
    jieba.add_word(w)


def seg(text: str) -> str:
    return " ".join(t for t in jieba.cut(text) if t.strip())


def split_text(text, size=CHUNK, overlap=OVERLAP):
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def build():
    rows = []
    for line in (HERE / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for ch in split_text(rec["text"]):
            rows.append((rec["title"], rec["region"], rec["version"], rec["url"], seg(ch), ch))

    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.execute("CREATE VIRTUAL TABLE chunks USING fts5("
                "title, region, version, url UNINDEXED, seg, raw UNINDEXED, tokenize='unicode61')")
    con.executemany("INSERT INTO chunks(title,region,version,url,seg,raw) VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    print(f"建库完成：{DB}\n  记录(页)数 {len({r[0] for r in rows})}｜切块数 {len(rows)}")
    con.close()


def query(q, k=5):
    toks = [t for t in jieba.cut(q) if t.strip()]
    match = " OR ".join(f'"{t}"' for t in toks)
    con = sqlite3.connect(DB)
    sql = ("SELECT title, region, version, url, raw FROM chunks "
           "WHERE chunks MATCH ? ORDER BY rank LIMIT ?")
    for row in con.execute(sql, (match, k)):
        raw = row[4]
        pos = raw.find(q[:2]) if len(q) >= 2 else -1
        snip = raw[max(0, pos - 30): pos + 90].replace("\n", " ") if pos >= 0 else raw[:120].replace("\n", " ")
        print(f"\n【{row[0]}】{row[1]} {row[2]}\n  …{snip}…\n  {row[3]}")
    con.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
    else:
        build()
