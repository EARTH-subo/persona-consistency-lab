# -*- coding: utf-8 -*-
"""抓取姬子相关任务的全文，清洗后存为语料 JSONL。

输入：rag/himeko_tasks.json（上一步生成的 80 页清单）
输出：rag/corpus.jsonl（每行 = {title, region, type, version, url, text}）
用法：python build_himeko_corpus.py
"""
import json
import re
import time
from pathlib import Path

import requests

API = "https://wiki.biligame.com/sr/api.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36"}
HERE = Path(__file__).resolve().parent
EXTRA_TITLES = ["姬子", "姬子/语音", "姬子/资料", "姬子•启行", "姬子·启行"]


def api(params):
    p = dict(params)
    p["format"] = "json"
    for _ in range(3):
        try:
            return requests.get(API, params=p, headers=HEADERS, timeout=25).json()
        except Exception:
            time.sleep(1.5)
    return {}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def clean(wikitext: str) -> str:
    t = wikitext
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<br\s*/?>", "\n", t)
    # {{...}} 两轮去模板（含参数）
    for _ in range(3):
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)   # [[a|b]] -> b
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)             # [[a]] -> a
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)  # [url text] -> text
    t = t.replace("'''", "").replace("''", "")
    t = re.sub(r"<[^>]+>", "", t)
    lines = []
    for ln in t.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(("|", "}", "{", "=")) and "：" not in ln and ":" not in ln:
            continue
        lines.append(ln)
    return "\n".join(lines)


def main():
    tasks = json.loads((HERE / "himeko_tasks.json").read_text(encoding="utf-8"))
    meta = {t["title"]: t for t in tasks}
    titles = [t["title"] for t in tasks] + EXTRA_TITLES

    out_path = HERE / "corpus.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for batch in chunks(titles, 50):
            r = api({"action": "query", "prop": "revisions", "rvprop": "content",
                     "rvslots": "main", "titles": "|".join(batch)})
            for pg in r.get("query", {}).get("pages", {}).values():
                if "revisions" not in pg:
                    continue
                raw = pg["revisions"][0]["slots"]["main"]["*"]
                m = meta.get(pg["title"], {})
                rec = {
                    "title": pg["title"],
                    "region": m.get("region", ""),
                    "type": m.get("type", ""),
                    "version": m.get("version", ""),
                    "url": m.get("url") or f"https://wiki.biligame.com/sr/{requests.utils.quote(pg['title'])}",
                    "text": clean(raw),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            print(f"batch done, records={n}")
            time.sleep(0.4)
    print("saved:", out_path, "records:", n)


if __name__ == "__main__":
    main()
