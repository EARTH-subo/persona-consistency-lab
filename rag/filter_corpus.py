# -*- coding: utf-8 -*-
"""过滤 corpus.jsonl：丢掉清洗后过短（<200 字）的噪声页，并同步更新 tasks 清单。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN = 200

recs = [json.loads(l) for l in (HERE / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
keep = [r for r in recs if len(r["text"]) >= MIN]
(HERE / "corpus.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in keep), encoding="utf-8")

tasks = [{k: v for k, v in r.items() if k != "text"} for r in keep]
tasks.sort(key=lambda x: (x["version"], x["title"]))
(HERE / "himeko_tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")

lines = [f"# 姬子相关页面清单（自动抓取，保留 {len(keep)} 页）", "",
         "> 来源：wiki.biligame.com/sr（官方内容转录镜像）｜三路合并：任务分类 + 全文搜索 + 指定页。",
         "> 已过滤清洗后 <200 字的导航/模板噪声页。字段：页面／地区／类型／版本／姬子次数／字数／链接。", "",
         "| 页面 | 地区 | 类型 | 版本 | 姬子次数 | 字数 | 链接 |", "|---|---|---|---|---|---|---|"]
for h in tasks:
    lines.append(f"| {h['title']} | {h['region']} | {h['type']} | {h['version']} | "
                 f"{h['himeko_mentions']} | {h['text_len']} | [wiki]({h['url']}) |")
(HERE / "姬子相关任务清单.md").write_text("\n".join(lines), encoding="utf-8")
print(f"过滤完成：保留 {len(keep)} / {len(recs)} 页（丢掉 {len(recs)-len(keep)} 个短噪声页）")
