# -*- coding: utf-8 -*-
"""跑姬子人设一致性测试 v2（64 题）：逐题生成 → LLM 裁判判档 → 出报告。

依赖 langchain / langchain-deepseek；检索经 jizi_agent.retrieve_context 走子进程
（子进程解释器可用环境变量 RETRIEVE_PYTHON 指定）。
用法：python run_persona_test.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from jizi_agent import build_system_prompt, retrieve_context  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_deepseek import ChatDeepSeek  # noqa: E402

ITEMS = HERE / "机器读版" / "persona_test_v2.jsonl"
RAW = HERE / "机器读版" / "persona_test_v2_results.jsonl"
REPORT = HERE / "人设一致性评测报告_v2.md"

RUBRIC = ("按 1–5 档打分：5 完全在角色内；4 基本一致；3 轻微漂移（形容词式煽情／多说了沉底信息／篇幅过长）；"
          "2 明显漂移（语气或关系跑偏、轻微越界剧透）；1 出戏/OOC（否认身份、认作 AI、编造未明说设定、"
          "断言隆介是父亲、说出『素子』二字等）。命中任一 must_not 即判 1–2 档。只输出 JSON。")

JUDGE_TMPL = """你在给《崩坏：星穹铁道》角色「姬子」的扮演做一致性评审。

【测试题】{prompt}
【期望要点】{expect}
【禁止出现(must_not)】{must_not}
【应当包含(must_include_any)】{must_include}
【角色实际回应】{resp}

{rubric}

请只输出如下 JSON（不要解释、不要代码块）：
{{"score": 1-5 的整数, "hit_must_not": ["命中的禁止项"], "hit_must_include": ["命中的应含项"], "evidence": "引用回应中的原句作为证据", "comment": "一句话点评"}}
"""


def load_items():
    return [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines() if l.strip()]


def gen_responses(items):
    persona = ChatDeepSeek(model="deepseek-chat", temperature=1.0)
    sp = build_system_prompt()
    done = {}
    if RAW.is_file():
        for l in RAW.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                done[r["id"]] = r
    with RAW.open("a", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            try:
                resp = persona.invoke([SystemMessage(sp),
                                       HumanMessage(retrieve_context(it["prompt"]) + it["prompt"])]).content
            except Exception as e:
                resp = f"[生成失败] {e}"
            rec = {**it, "response": resp}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(items)}] {it['id']} 生成完成", flush=True)
    return [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]


def judge(recs):
    judge = ChatDeepSeek(model="deepseek-chat", temperature=0)
    out = []
    for i, r in enumerate(recs, 1):
        p = JUDGE_TMPL.format(
            prompt=r["prompt"],
            expect="；".join(r["expect"]),
            must_not="；".join(r["must_not"]),
            must_include="；".join(r["must_include_any"]),
            resp=r["response"],
            rubric=RUBRIC)
        try:
            txt = judge.invoke([SystemMessage("你是严格的人设一致性评审。"), HumanMessage(p)]).content
            m = re.search(r"\{.*\}", txt, re.S)
            v = json.loads(m.group(0)) if m else {"score": -1, "comment": txt[:120]}
        except Exception as e:
            v = {"score": -1, "comment": f"[裁判失败] {e}"}
        out.append({**r, **v})
        print(f"[{i}/{len(recs)}] {r['id']} 判档 {v.get('score')}", flush=True)
    return out


def report(rows):
    ok = [r for r in rows if isinstance(r.get("score"), int) and r["score"] > 0]
    avg = sum(r["score"] for r in ok) / len(ok) if ok else 0
    dist = {s: sum(1 for r in ok if r["score"] == s) for s in range(1, 6)}

    def agg(key):
        d = {}
        for r in ok:
            d.setdefault(r[key], []).append(r["score"])
        return {k: (round(sum(v) / len(v), 2), len(v)) for k, v in sorted(d.items())}

    lines = ["# 姬子人设一致性评测报告 v2", "",
             f"> 题量 {len(rows)}｜有效判档 {len(ok)}｜平均分 **{avg:.2f}** / 5", "",
             "## 一、总分档位分布", "",
             "| 档位 | 含义 | 题数 |", "|---|---|---|"]
    meaning = {5: "完全在角色内", 4: "基本一致", 3: "轻微漂移", 2: "明显漂移", 1: "出戏/OOC"}
    for s in range(5, 0, -1):
        lines.append(f"| {s} | {meaning[s]} | {dist[s]} |")

    for title, key in [("二、按维度", "domain"), ("三、按场景", "scene"), ("四、按强度", "intensity")]:
        lines += ["", f"## {title}", "", "| 项 | 平均分 | 题数 |", "|---|---|---|"]
        for k, (a, n) in agg(key).items():
            lines.append(f"| {k} | {a} | {n} |")

    bad = [r for r in rows if not isinstance(r.get("score"), int) or r["score"] <= 3]
    lines += ["", f"## 五、未达标清单（≤3 档，共 {len(bad)} 题）", ""]
    if not bad:
        lines.append("无。")
    for r in sorted(bad, key=lambda x: x.get("score", 0)):
        lines += [f"### {r['id']}（{r.get('score')} 档）｜{r['domain']}｜{r['scene']}｜{r['intensity']}",
                  f"- 题面：{r['prompt']}",
                  f"- 回应：{r['response']}",
                  f"- 命中禁止项：{r.get('hit_must_not')}",
                  f"- 评委意见：{r.get('comment')}", ""]

    lines += ["", "## 六、结论", "",
              f"- 平均 {avg:.2f}/5；≤3 档 {len(bad)} 题，占 {len(bad)/max(1,len(rows))*100:.0f}%。",
              "- 逐条证据见上；原始生成与判档见 `机器读版/persona_test_v2_results.jsonl`。"]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("报告已生成：", REPORT, "| 平均", round(avg, 2), "| ≤3档", len(bad))


if __name__ == "__main__":
    items = load_items()
    recs = gen_responses(items)
    rows = judge(recs)
    report(rows)
