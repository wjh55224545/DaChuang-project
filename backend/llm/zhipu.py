"""
智谱AI (GLM) 客户端
===================
用于反馈模块的智能通知内容生成。
通过智谱AI API调用GLM-4模型，根据学生心理分析结果生成个性化的多渠道通知文本。

API文档: https://open.bigmodel.cn/dev/api/normal-model/glm-4
"""

from __future__ import annotations
import json
from functools import lru_cache
from typing import Optional
from openai import OpenAI

from backend.config import get_settings


@lru_cache
def get_zhipu_client() -> OpenAI:
    """获取智谱AI客户端（单例，兼容OpenAI接口）"""
    settings = get_settings()
    return OpenAI(
        api_key=settings.zhipu_api_key,
        base_url=settings.zhipu_base_url,
    )


def generate_feedback_content(
    student_name: str,
    severity: str,
    risk_reason: str,
    overall_score: float,
    risk_factors: list[str],
    suggestions: list[str],
    emotion_distribution: dict,
    class_name: str = "",
    student_code: str = "",
    model: Optional[str] = None,
) -> dict:
    """
    调用智谱AI生成个性化反馈通知内容。

    Args:
        student_name: 学生姓名
        severity: 风险等级 (green/yellow/red)
        risk_reason: 风险原因描述
        overall_score: 综合心理健康评分
        risk_factors: 风险因素列表
        suggestions: 建议列表
        emotion_distribution: 情绪分布
        class_name: 班级名称
        student_code: 学号
        model: 模型名称，默认使用配置值

    Returns:
        dict: 包含各渠道通知内容的字典
            {
                "title": str,
                "dashboard": str,
                "app_push": str,
                "wechat_teacher": str,
                "wechat_parent": str,
                "sms_parent": str,
                "email_psychologist": str,
            }
    """
    settings = get_settings()
    client = get_zhipu_client()

    # 构建风险因素文本
    risk_factors_text = "；".join(risk_factors) if risk_factors else "无明显风险因素"

    # 构建建议文本
    suggestion_texts = []
    for s in suggestions:
        if isinstance(s, dict):
            suggestion_texts.append(s.get("content", ""))
        else:
            suggestion_texts.append(str(s))
    suggestions_text = "；".join(suggestion_texts) if suggestion_texts else "建议持续关注"

    # 构建情绪分布文本
    emotion_text = ", ".join([f"{k}: {v}次" for k, v in emotion_distribution.items()]) if emotion_distribution else "无数据"

    # 等级说明
    severity_desc = {
        "green": "低风险（绿色）— 学生情绪状态良好，继续保持常规关注",
        "yellow": "中风险（黄色）— 学生情绪有波动，需要班主任和心理委员加强关注",
        "red": "高风险（红色）— 学生情绪状态异常，需要立即启动干预流程",
    }

    system_prompt = """你是一个专业的心理健康通知生成助手，服务于学校心理健康监测系统"心镜智能体"。
你的任务是根据学生的心理健康分析数据，生成不同渠道的通知内容。

要求：
1. 语言专业、温暖、有同理心，不使用可能引起恐慌的措辞
2. 根据风险等级调整语气：绿色用鼓励语气，黄色用关切语气，红色用紧急但镇定的语气
3. 不同渠道的内容应该适配其特点：
   - dashboard（看板）：面向心理教师，简洁专业，突出关键数据
   - app_push（APP推送）：面向学生本人，鼓励为主，保护自尊心
   - wechat_teacher（微信班主任）：面向班主任，提供可操作的建议
   - wechat_parent（微信家长）：面向家长，温和告知，避免引起过度焦虑
   - sms_parent（短信家长）：精简版家长通知，50字以内
   - email_psychologist（邮件心理教师）：详细专业报告，包含完整数据和干预建议
4. 保护学生隐私，不要在非必要渠道暴露具体评分数据
5. 所有内容使用中文"""

    user_prompt = f"""请根据以下学生心理健康分析数据，生成各渠道的通知内容：

【学生信息】
姓名：{student_name}
班级：{class_name or '未知班级'}
学号：{student_code or '未知学号'}

【评估结果】
风险等级：{severity_desc.get(severity, severity)}
综合心理健康评分：{overall_score:.2f}/1.00
风险原因：{risk_reason}

【风险因素】
{risk_factors_text}

【改进建议】
{suggestions_text}

【近期情绪分布】
{emotion_text}

请以JSON格式返回各渠道通知内容，格式如下：
{{
  "title": "通知标题（含emoji）",
  "dashboard": "看板通知内容（面向心理教师）",
  "app_push": "APP推送内容（面向学生）",
  "wechat_teacher": "微信通知内容（面向班主任）",
  "wechat_parent": "微信通知内容（面向家长，仅黄色/红色需要）",
  "sms_parent": "短信通知内容（面向家长，仅红色需要，50字内）",
  "email_psychologist": "邮件正文（面向心理教师，仅红色需要，包含完整评估）"
}}

只返回JSON，不要包含其他文字。"""

    try:
        response = client.chat.completions.create(
            model=model or settings.zhipu_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("智谱AI返回空内容")

        result = json.loads(content)
        return result

    except json.JSONDecodeError:
        # JSON解析失败时返回原始文本
        raw_text = response.choices[0].message.content if 'response' in dir() else ""
        return {
            "title": f"心理健康通知 — {student_name}",
            "dashboard": raw_text or f"学生{student_name}，风险等级{severity}，综合评分{overall_score:.2f}",
            "app_push": f"同学你好，{'今日情绪状态良好，继续保持！' if severity == 'green' else '近期情绪有些波动，记得照顾好自己。'}",
            "wechat_teacher": raw_text or f"学生{student_name}需要关注，风险等级{severity}",
            "wechat_parent": "" if severity == "green" else raw_text or "",
            "sms_parent": "" if severity != "red" else f"【心镜智能】{student_name}家长，您的孩子近期需要关注，学校心理教师将尽快与您联系。",
            "email_psychologist": "" if severity != "red" else raw_text or "",
        }

    except Exception as e:
        raise RuntimeError(f"智谱AI通知生成失败: {str(e)}") from e


def generate_simple_notification(
    student_name: str,
    severity: str,
    content: str,
    model: Optional[str] = None,
) -> str:
    """
    快速生成简短通知内容（用于非紧急场景）。

    Args:
        student_name: 学生姓名
        severity: 风险等级
        content: 原始预警内容
        model: 模型名称

    Returns:
        str: 生成的通知文本
    """
    settings = get_settings()
    client = get_zhipu_client()

    severity_desc = {
        "green": "正常状态",
        "yellow": "需要关注",
        "red": "紧急状态",
    }

    prompt = f"""请为学校心理健康监测系统生成一条简短的通知消息。
学生：{student_name}
状态：{severity_desc.get(severity, severity)}
原始信息：{content}

要求：语气温暖专业，30-80字，一句话即可。只返回通知文本，不要其他内容。"""

    try:
        response = client.chat.completions.create(
            model=model or settings.zhipu_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=256,
        )
        text = response.choices[0].message.content
        return text.strip() if text else content

    except Exception:
        return content
