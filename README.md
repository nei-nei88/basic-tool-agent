# Basic Tool Agent

这是我第二周完成的 LLM Agent 入门项目，目标是学习 Prompt、结构化输出和 Tool Calling。

本项目基于 DeepSeek API 和 OpenAI 兼容 SDK，实现了一个最小可运行的工具调用 Agent。用户输入一个任务后，模型会判断任务类型，并决定是否调用工具函数，最后输出结构化 JSON 结果。

## 项目目标

本项目的目标是完成一个最小 Agent 闭环：

```text
用户输入任务
↓
模型判断任务类型
↓
选择是否调用工具
↓
Python 执行对应工具
↓
模型根据工具结果整理回答
↓
输出结构化 JSON