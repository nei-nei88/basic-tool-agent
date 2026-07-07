import argparse
import ast
import json
import operator
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI


# =========================
# 1. 读取环境变量
# =========================

def create_client() -> tuple[OpenAI, str]:
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    if not api_key:
        raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60,
        max_retries=3,
    )

    return client, model


# =========================
# 2. 工具一：安全计算器
# 不要直接用 eval，危险。
# =========================

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def safe_eval_math(expression: str) -> float:
    """
    安全计算简单数学表达式。
    支持 + - * / ** % 和括号。
    """

    def eval_node(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("只支持数字。")

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"不支持的运算符：{op_type}")

            return ALLOWED_OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"不支持的一元运算符：{op_type}")

            return ALLOWED_OPERATORS[op_type](operand)

        raise ValueError("表达式中包含不支持的内容。")

    tree = ast.parse(expression, mode="eval")
    return eval_node(tree.body)


def calculator(expression: str) -> Dict[str, Any]:
    result = safe_eval_math(expression)
    return {
        "tool": "calculator",
        "expression": expression,
        "result": result,
    }


# =========================
# 3. 工具二：文本总结器
# 这里复用第一周能力：工具内部再次调用 LLM 做总结
# =========================

def summarizer(text: str, client: OpenAI, model: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个严谨的中文文本总结助手。不要编造，只总结用户提供的内容。",
            },
            {
                "role": "user",
                "content": f"""
请总结下面文本，输出 3 个部分：

1. 一句话总结
2. 关键要点，最多 5 条
3. 适合后续追问的问题，最多 3 个

文本：
{text}
""",
            },
        ],
        stream=False,
    )

    summary = response.choices[0].message.content

    return {
        "tool": "summarizer",
        "summary": summary,
    }


# =========================
# 4. 工具三：学习计划器
# 先做一个本地规则版，不额外调用 LLM
# =========================

def planner(goal: str, days: int = 3) -> Dict[str, Any]:
    if days <= 0:
        raise ValueError("days 必须大于 0。")

    plan = []

    for day in range(1, days + 1):
        plan.append(
            {
                "day": day,
                "task": f"围绕目标「{goal}」完成第 {day} 阶段学习与实践。",
                "output": f"产出第 {day} 天的笔记或代码提交。",
            }
        )

    return {
        "tool": "planner",
        "goal": goal,
        "days": days,
        "plan": plan,
    }


# =========================
# 5. 告诉模型有哪些工具
# 这部分不是 Python 函数本身，而是工具说明书
# =========================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "用于计算数学表达式，比如 128 * 37 + 56。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，只包含数字和基础运算符。",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarizer",
            "description": "用于总结一段较长文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要总结的原始文本。",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "planner",
            "description": "用于把一个学习目标或任务目标拆成多天计划。",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "用户想完成的目标。",
                    },
                    "days": {
                        "type": "integer",
                        "description": "计划天数，默认 3 天。",
                    },
                },
                "required": ["goal"],
            },
        },
    },
]


# =========================
# 6. 根据模型返回的 tool_call 执行真正的 Python 函数
# =========================

def run_tool(tool_name: str, arguments: Dict[str, Any], client: OpenAI, model: str) -> Dict[str, Any]:
    try:
        if tool_name == "calculator":
            return calculator(arguments["expression"])

        if tool_name == "summarizer":
            return summarizer(arguments["text"], client, model)

        if tool_name == "planner":
            return planner(
                goal=arguments["goal"],
                days=arguments.get("days", 3),
            )

        return {
            "error": f"未知工具：{tool_name}",
        }

    except Exception as e:
        return {
            "tool": tool_name,
            "error": str(e),
        }


# =========================
# 7. Agent 主流程
# =========================

def run_agent(user_task: str) -> Dict[str, Any]:
    client, model = create_client()

    system_prompt = """
你是一个 basic-tool-agent。

你可以根据用户任务选择工具：
- 数学计算任务：调用 calculator
- 长文本总结任务：调用 summarizer
- 学习计划、任务拆解、日程安排：调用 planner

如果需要工具，请调用最合适的工具。
如果不需要工具，可以直接回答。

最终回答必须是 json，格式如下：

{
  "task_type": "calculation | summarization | planning | general",
  "used_tool": true,
  "tool_name": "calculator | summarizer | planner | none",
  "result": "给用户看的最终结果",
  "reason": "简单解释为什么这样处理"
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task},
    ]

    # 第一次请求：让模型判断是否需要调用工具
    first_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=False,
    )

    assistant_message = first_response.choices[0].message

    # 如果模型没有调用工具，直接要求它输出 JSON
    if not assistant_message.tool_calls:
        final_response = client.chat.completions.create(
            model=model,
            messages=[
                *messages,
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                },
                {
                    "role": "user",
                    "content": "请把上面的回答整理成指定 json 格式。",
                },
            ],
            response_format={"type": "json_object"},
            stream=False,
        )

        return json.loads(final_response.choices[0].message.content)

    # 如果模型调用了工具：
    # 1. 把模型的 tool_call 消息加入对话
    messages.append(assistant_message.model_dump(exclude_none=True))

    tool_results = []

    # 2. Python 真正执行工具函数
    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        tool_result = run_tool(tool_name, arguments, client, model)
        tool_results.append(tool_result)

        # 3. 把工具结果交回模型
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )

    # 4. 第二次请求：让模型根据工具结果整理最终 JSON
    final_response = client.chat.completions.create(
        model=model,
        messages=[
            *messages,
            {
                "role": "user",
                "content": "请根据工具结果，输出最终 json。不要输出 markdown，不要输出代码块。",
            },
        ],
        response_format={"type": "json_object"},
        stream=False,
    )
    
    final_content = final_response.choices[0].message.content

    return json.loads(final_content)


# =========================
# 8. 命令行入口
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="A basic LLM tool-calling agent.")
    parser.add_argument(
        "task",
        type=str,
        help="用户任务，例如：帮我计算 128 * 37 + 56",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_agent(args.task)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()