# 智能体 Prompt 与解析模块使用说明

## 模块概述
本模块为“心理状态监测智能体”提供 ReAct 推理所需的完整 System Prompt 和输出解析器。
智能体基于 LangChain（或自定义循环）调用大语言模型，通过本模块的 Prompt 指导 LLM 按格式输出，
并通过解析器将自然语言转化为可执行的动作。

## 文件说明
| 文件                       | 功能                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `prompts/system_prompt.py` | 定义 SYSTEM_PROMPT 字符串，内含工具描述、Few-shot 示例       |
| `parsers/output_parser.py` | 实现 `parse_agent_output()` 函数，从 LLM 原始文本中提取动作和参数 |
| `tests/test_prompt.py`     | 自动化测试脚本，验证核心解析逻辑的正确性                     |

## 集成方法
### 方式一：文本 ReAct 模式（适用所有 LLM）
```python
from prompts.system_prompt import SYSTEM_PROMPT
from parsers.output_parser import parse_agent_output

# 在 Agent 主循环中构建 messages
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_input}
]

llm_response = llm.chat(messages)  # 获取原始文本
parsed = parse_agent_output(llm_response)
if "action" in parsed:
    # 执行工具，将 Observation 加入上下文
    observation = execute_tool(parsed["action"], parsed["action_input"])
    messages.append({"role": "assistant", "content": llm_response})
    messages.append({"role": "user", "content": f"Observation: {observation}"})
elif "final_answer" in parsed:
    # 返回最终答案
    return parsed["final_answer"]
else:
    # 错误处理，可将错误信息反馈给 LLM 重试
    messages.append({"role": "user", "content": f"解析错误：{parsed['error']}，请按格式重新输出。"})