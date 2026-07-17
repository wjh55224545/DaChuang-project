"""
ReAct 输出解析器
负责从 LLM 生成的原始文本中提取 Action 和 Action Input，
提供稳定的结构化动作指令，并针对常见格式错误进行容错处理。
"""

import re
import json
from typing import Dict, Any, Optional


def parse_agent_output(text: str) -> Dict[str, Any]:
    """
    解析 LLM 的 ReAct 输出，返回结构化动作。

    参数:
        text (str): LLM 的完整输出文本

    返回:
        dict: 包含以下键之一：
            - {'action': str, 'action_input': dict}  正常解析
            - {'final_answer': str}                   最终回答
            - {'error': str, 'raw': str}              解析失败
    """
    # 1. 检查是否为最终回答
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if final_match:
        # 如果有 Final Answer 且在 Action 之前，直接返回最终答案
        final_pos = final_match.start()
        # 查找最后一个 Action 关键字的位置，避免 Final Answer 出现在 Action 前面误判
        action_matches = list(re.finditer(r"Action:", text, re.IGNORECASE))
        if not action_matches or all(m.start() > final_pos for m in action_matches):
            return {"final_answer": final_match.group(1).strip()}

    # 2. 提取 Action 名称
    action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    if not action_match:
        # 尝试从可能混合的格式中提取
        return {"error": "未能找到 Action 字段", "raw": text}
    action = action_match.group(1).strip()

    # 3. 提取 Action Input（JSON 格式）
    # 匹配紧随 Action Input: 后的大括号内容，允许跨行
    input_match = re.search(r"Action Input:\s*(\{.*?\})\s*(?:$|Observation:|Thought:|Action:|Final Answer:)", 
                            text, re.DOTALL | re.IGNORECASE)
    if input_match:
        json_str = input_match.group(1)
    else:
        # 宽松匹配：找到最后一个 JSON 对象
        json_match = re.search(r"\{[^{}]*\}", text)
        if json_match:
            json_str = json_match.group(0)
        else:
            # 尝试处理非 JSON 输入（如 image_path="xxx"）
            param_match = re.search(r"image_path\s*=\s*['\"]([^'\"]+)['\"]", text)
            if param_match and action in ("多模态情绪识别", "EmotionDetector"):
                return {"action": action, "action_input": {"image_path": param_match.group(1)}}
            return {"error": "Action Input 缺失或格式错误", "raw": text}

    # 4. 解析 JSON
    try:
        action_input = json.loads(json_str)
    except json.JSONDecodeError:
        # 常见修复：单引号转双引号
        try:
            fixed = json_str.replace("'", '"')
            action_input = json.loads(fixed)
        except:
            return {"error": "Action Input 不是合法 JSON", "raw": text, "json_str": json_str}

    # 5. 工具名称白名单校验
    allowed_actions = [
        "多模态情绪识别", "华为云OBS持久化", "时序心理分析", "多渠道反馈",
        "Search", "Calculator", "EmotionDetector",  # 向后兼容旧工具名
    ]
    if action not in allowed_actions:
        return {"error": f"未定义的工具: {action}", "raw": text}

    return {"action": action, "action_input": action_input}


# 可选：为支持 function calling 的模型提供工具定义
def get_functions_definitions():
    """
    返回 OpenAI/兼容接口的 functions 参数定义。
    """
    return [
        {
            "name": "Search",
            "description": "当需要获取现实世界信息时使用，如知识查询、新闻等",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "Calculator",
            "description": "当需要进行数学计算时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '3*5'"
                    }
                },
                "required": ["expression"]
            }
        },
        {
            "name": "EmotionDetector",
            "description": "当用户上传图片并要求分析情绪时使用，需提供图片文件路径",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "图片文件的绝对路径或相对路径"
                    }
                },
                "required": ["image_path"]
            }
        }
    ]