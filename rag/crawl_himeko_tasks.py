# -*- coding: utf-8 -*-
"""姬子语料抓取 v2：三路合并标题 → 抓全文 → 清洗 → 输出 tasks 清单 + corpus.jsonl。

三路来源：
  A. 任务分类全量（开拓任务/同行任务/开拓续闻/冒险任务/活动任务/终末任务/日常任务/任务）
  B. MediaWiki 全文搜索 "姬子"（namespace 0）
  C. 指定页（姬子相关角色页/语音页）
用法：python crawl_himeko_tasks.py
"""
import json
import re
import time
from pathlib import Path

import requests

API = "https://wiki.biligame.com/sr/api.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36"}
CATS = ["Category:开拓任务", "Category:同行任务", "Category:开拓续闻", "Category:冒险任务",
        "Category:活动任务", "Category:终末任务", "Category:日常任务", "Category:任务"]
EXTRA = ["姬子", "姬子/语音", "姬子/资料", "姬子/角色故事", "姬子•启行", "姬子·启行",
         "姬子•启行/语音", "姬子·启行/语音"]
HERE = Path(__file__).resolve().parent
OUT_TASKS = HERE / "himeko_tasks.json"
OUT_CORPUS = HERE / "corpus.jsonl"


_LAST = [0.0]


def api(params):
    p = dict(params); p["format"] = "json"
    for i in range(5):
        try:
            gap = time.time() - _LAST[0]
            if gap < 0.4:
                time.sleep(0.4 - gap)
            r = requests.get(API, params=p, headers=HEADERS, timeout=30)
            _LAST[0] = time.time()
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return {}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def field(txt, key):
    m = re.search(rf"{key}\s*=\s*([^\n|}}]+)", txt)
    return m.group(1).strip() if m else ""


def clean(wikitext):
    t = wikitext
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<br\s*/?>", "\n", t)
    for _ in range(3):
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)
    t = t.replace("'''", "").replace("''", "")
    t = re.sub(r"<[^>]+>", "", t)
    keep = []
    for ln in t.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln[0] in "|{}=}" and "：" not in ln and ":" not in ln:
            continue
        keep.append(ln)
    return "\n".join(keep)


def gather_titles():
    titles = set()
    for cat in CATS:
        got = 0
        cont = {}
        while True:
            p = {"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": 500}
            p.update(cont)
            r = api(p)
            ms = r.get("query", {}).get("categorymembers", [])
            for m in ms:
                titles.add(m["title"])
            got += len(ms)
            cont = r.get("continue") or {}
            if not cont:
                break
        print(f"  分类 {cat}: {got}", flush=True)
    # B. 全文搜索
    off, sgot = 0, 0
    while True:
        r = api({"action": "query", "list": "search", "srsearch": "姬子",
                 "srnamespace": 0, "srlimit": 500, "sroffset": off})
        hits = r.get("query", {}).get("search", [])
        for h in hits:
            titles.add(h["title"])
        sgot += len(hits)
        off = r.get("continue", {}).get("sroffset")
        if not off:
            break
    print(f"  搜索 '姬子': {sgot}", flush=True)
    titles.update(EXTRA)
    return titles


def main():
    titles = sorted(gather_titles())
    print("候选标题数：", len(titles))
    tasks, n = [], 0
    with OUT_CORPUS.open("w", encoding="utf-8") as f:
        for batch in chunks(titles, 50):
            r = api({"action": "query", "prop": "revisions", "rvprop": "content",
                     "rvslots": "main", "titles": "|".join(batch)})
            for pg in r.get("query", {}).get("pages", {}).values():
                if "revisions" not in pg:
                    continue
                raw = pg["revisions"][0]["slots"]["main"]["*"]
                if "姬子" not in raw:
                    continue
                txt = clean(raw)
                rec = {"title": pg["title"], "region": field(raw, "任务地区"),
                       "type": field(raw, "任务类型"), "version": field(raw, "所属版本"),
                       "himeko_mentions": raw.count("姬子"), "text_len": len(txt),
                       "url": f"https://wiki.biligame.com/sr/{requests.utils.quote(pg['title'])}"}
                tasks.append(rec)
                f.write(json.dumps({**rec, "text": txt}, ensure_ascii=False) + "\n")
                n += 1
            print("batch done, kept =", n, flush=True)
            time.sleep(0.4)
    tasks.sort(key=lambda x: (x["version"], x["title"]))
    OUT_TASKS.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    short = [t for t in tasks if t["text_len"] < 200]
    print(f"含姬子页数：{n}｜清洗后过短(<200字)的：{len(short)}")
    for s in short[:20]:
        print("  短页：", s["title"], s["text_len"])


if __name__ == "__main__":
    main()
