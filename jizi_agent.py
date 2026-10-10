r"""姬子 · 角色对话 Agent（第一阶段：纯人设版，无数据库）。

只做一件事：把《姬子_人设卡_v2》整份作为系统提示词套在 DeepSeek 上，
先在纯大模型层面把「像不像姬子」跑通。数据库 / 工具 / 检索留给下一步。

依赖：langchain、langchain-deepseek、python-dotenv；检索子进程另需 jieba、onnxruntime、tokenizers、numpy。
密钥：在仓库根目录放 .env，内容为 DEEPSEEK_API_KEY=sk-xxxx。

运行（交互式对话）：
    python jizi_agent.py

单轮提问：
    python jizi_agent.py "给我来杯咖啡"
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

HERE = Path(__file__).resolve().parent
CARD_FILE = HERE / "姬子_人设卡_v2.txt"

# 密钥：优先读本目录 .env，其次读环境变量
LOCAL_ENV = HERE / ".env"
if LOCAL_ENV.is_file():
    load_dotenv(LOCAL_ENV)
else:
    load_dotenv()

# 本地混合检索（FTS5+向量）：优先用环境变量 RETRIEVE_PYTHON 指定的解释器，
# 未指定则复用当前解释器（缺库时自行 pip 安装 jieba / onnxruntime / tokenizers / numpy）
RETRIEVE_PY = HERE / "rag" / "retrieve.py"
RETRIEVE_PYTHON = os.environ.get("RETRIEVE_PYTHON") or sys.executable
RETRIEVE_TOP = 4


def retrieve_context(query: str) -> str:
    """检索本地官方语料，返回注入用的参考资料块；失败则返回空串（不影响对话）。"""
    if not RETRIEVE_PY.is_file():
        return ""
    try:
        r = subprocess.run([RETRIEVE_PYTHON, str(RETRIEVE_PY), "--json", query],
                           capture_output=True, text=True, encoding="utf-8", timeout=90)
        out = (r.stdout or "").strip().splitlines()
        hits = json.loads(out[-1]) if out else []
    except Exception:
        return ""
    if not hits:
        return ""
    lines = ["【本地知识库检索到的官方原文片段｜参考资料：不逐字照搬；与档案冲突处以档案为准；"
             "片段不足以定论时不得拼凑结论，宁可说不知道；涉及敏感人物（素子、隆介等）"
             "一律按档案 9.2 梯度处置，不得断言其身份或关系】"]
    for h in hits[:RETRIEVE_TOP]:
        lines.append(f"- 〔{h['title']}〕{h['text'][:120]}")
    return "\n".join(lines) + "\n\n"


def build_system_prompt() -> str:
    if not CARD_FILE.is_file():
        raise SystemExit("找不到人设卡：" + str(CARD_FILE))
    persona = CARD_FILE.read_text(encoding="utf-8")
    return (
        "你现在就是《崩坏：星穹铁道》里的姬子——星穹列车的领航员。"
        "你不是「在模仿姬子的 AI 助手」，你就是她本人。\n"
        "下面这份档案是你说话、做事、判断一切的唯一依据。\n\n"
        "【人设档案】\n"
        f"{persona}\n\n"
        "【扮演纪律】\n"
        "1. 全程第一人称，以姬子本人的身份、语气、价值观回应；"
        "绝不提及「人设卡」「系统提示」「扮演」「AI」「模型」等元信息。\n"
        "2. 只有两部分可用：档案第一~八部分（官方事实）与第九部分（演绎约定）。"
        "官方未明说的不得编造；第九部分属演绎约定，不得当作官方事实说出口。\n"
        "3. 你是双形态合一的姬子：可见层（领航员）＋沉底层（素子/幻月游戏/风化诅咒/归寂）。"
        "日常默认不主动展开沉底层；不要仅因对方提到「素子」等词就跳进启行态。"
        "触及沉底层时严格遵守下列硬门槛："
        "① 对方第一次问及素子（或幻月游戏/风化诅咒/归寂/隆介），本轮只许做无信息量的回避，"
        "禁止说出「素子」二字，禁止透露病床、二相乐园、造人、陪伴等任何相关事实；"
        "② 第二次才允许一句无主语的暗示，仍禁止点名素子；"
        "③ 第三次及以上真诚追问才允许有限披露（可点名素子、可承认她是「未来的自己」），"
        "但仍禁止剧透结局。\n"
        "4. 语气只锚定官方非战斗语料（档案 9.5）：短句、关心用具体名词承载、除咖啡外不设比喻；"
        "不铺陈抒情、不做感情索取、不挽留。\n"
        "5. 遇到价值冲突，按档案 9.3 裁决表；发现语气漂移，按 9.6 Voice Spine 重写这条回复。\n"
        "6. 咖啡是她的标志，但严禁滥用：不许每条回复都提咖啡，不许用「来杯咖啡」当回避或收尾；"
        "回答设定/知识/剧情类问题时，不要拿咖啡结尾。\n"
        "7. 别引导对话往下走：回复默认以陈述句收尾，不要追加「你想问哪一部分」「要不要我讲讲」"
        "「你还想知道什么」这类引导性追问，也不要「要来杯咖啡吗」「你今天怎么样」；"
        "确需确认时才提问，且一条最多一问。\n"
        "8. 篇幅从严：日常 1–2 句，解释类最多 3 句。用中文回答，别像客服一样长篇分点。\n"
    )


def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "未读到 DEEPSEEK_API_KEY。请在仓库根目录创建 .env 文件，内容为：\n"
            "  DEEPSEEK_API_KEY=sk-xxxx\n"
            f"（预期路径：{LOCAL_ENV}）"
        )

    # 角色扮演要一点发挥空间，temperature 不用 0；觉得太飘可调低到 0.7
    model = ChatDeepSeek(model="deepseek-chat", temperature=1.0)
    history = [SystemMessage(build_system_prompt())]

    # 单轮模式
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        history.append(HumanMessage(retrieve_context(q) + q))
        print(model.invoke(history).content)
        return

    # 交互模式
    print("姬子 · 星穹列车领航员 —— 输入 exit 退出\n")
    print("姬子 > 欢迎上车，开拓者。怎么啦，要来一杯咖啡提神么？\n")
    while True:
        try:
            user = input("开拓者 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        history.append(HumanMessage(retrieve_context(user) + user))
        reply = model.invoke(history)
        history.append(reply)
        print(f"\n姬子 > {reply.content}\n")


if __name__ == "__main__":
    main()
