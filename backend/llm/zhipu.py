"""
智谱AI (GLM) 客户端
===================
封装智谱AI API（兼容OpenAI接口）的调用逻辑，为心镜智能体提供以下能力：

1. generate_feedback_content()
   反馈模块：根据学生心理分析结果，生成个性化的多渠道智能通知文本。

2. generate_mental_health_analysis()  [新增]
   时序分析模块：基于数学事实数据（熵、趋势、恢复速度等），
   由GLM-4-Flash推理7项需专业判断的心理健康指标，并进行风险评估和建议生成。

3. generate_simple_notification()
   快速通知：生成简短的预警通知消息（非紧急场景）。

统一使用 settings.zhipu_api_key 和 settings.zhipu_base_url 配置。

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


def generate_mental_health_analysis(
    student_id: int,
    baseline: float,
    avg_emotion: float,
    variance: float,
    entropy: float,
    trend_slope: float,
    trend: str,
    abrupt_count: int,
    recovery_speed: float,
    recovery_count: int,
    avg_recovery_interval: str,
    positive_ratio: float,
    negative_ratio: float,
    emotion_distribution: dict,
    valence_series: list,
    arousal_series: list,
    record_count: int,
    analysis_window_days: int = 7,
    model: Optional[str] = None,
) -> dict:
    """
    调用智谱AI进行时序心理健康深度分析。

    基于数学计算得出的事实数据（熵、趋势、恢复速度等），让AI推理出
    需要专业判断的指标（稳定性指数、压力累积、综合评分等），并进行
    风险评估和建议生成。

    Args:
        student_id: 学生ID
        baseline: 历史情绪基线
        avg_emotion: 平均情绪分数
        variance: 情绪方差
        entropy: Shannon 熵值
        trend_slope: 趋势斜率
        trend: 趋势方向 (改善中/稳定/下降中)
        abrupt_count: 情绪突变次数
        recovery_speed: 恢复速度
        recovery_count: 恢复次数
        avg_recovery_interval: 平均恢复间隔
        positive_ratio: 积极情绪占比
        negative_ratio: 负面情绪占比
        emotion_distribution: 情绪分布 {标签: 频次}
        valence_series: Valence 时间序列
        arousal_series: Arousal 时间序列
        record_count: 有效记录数
        analysis_window_days: 分析窗口天数
        model: 模型名称

    Returns:
        dict: {
            "emotional_stability_index": float,     # ① 情绪稳定性指数
            "negative_emotion_accumulation": float, # ③ 负面情绪累积度
            "social_interaction_frequency": float,  # ④ 社交互动频次
            "arousal_abnormality_index": float,     # ⑥ 唤醒度异常指数
            "sleep_quality_prediction": float,      # ⑧ 睡眠质量预测
            "stress_accumulation_index": float,     # ⑨ 压力累积指数
            "overall_mental_health_score": float,   # ⑫ 综合心理健康评分
            "risk_level": str,                      # green/yellow/red
            "risk_reason": str,
            "reasoning_chain": list[str],
            "suggestions": list[dict],
            "risk_factors": list[str],
            "protective_factors": list[str],
            "detected_patterns": list[dict],
            "forecast": {"next_period_trend": str, "confidence": float, "key_uncertainty": str},
        }
    """
    settings = get_settings()
    client = get_zhipu_client()

    # 构建情绪分布文本
    emotion_text = ", ".join(
        [f"{k}: {v}次" for k, v in emotion_distribution.items()]
    ) if emotion_distribution else "无数据"

    # Valence 序列摘要（取最近10个点，避免token过长）
    if valence_series:
        valence_summary = ", ".join(f"{v:.2f}" for v in valence_series[-10:])
    else:
        valence_summary = "无数据"

    # Arousal 序列摘要
    if arousal_series:
        arousal_summary = ", ".join(f"{a:.2f}" for a in arousal_series[-10:])
    else:
        arousal_summary = "无数据"

    system_prompt = """你是一名学校心理健康评估专家，服务于"心镜智能体"心理健康监测系统。\n\n你的任务是根据学生情绪监测的客观统计数据，进行专业的心理健康评估。你需要：\n\n1. **计算7项指标**：基于提供的数学事实数据，运用专业知识给出以下指标值：\n   - 情绪稳定性指数（0-1，越高越稳定）\n   - 负面情绪累积度（0-1，越高负面越多）\n   - 社交互动频次（0-1，越高越活跃）\n   - 唤醒度异常指数（0-1，越高越异常）\n   - 睡眠质量预测（0-1，越高越好）\n   - 压力累积指数（0-1，越高压力越大）\n   - 综合心理健康评分（0-1，越高越健康）\n\n2. **风险评估**：\n   - 绿色(>=0.7)：情绪状态良好\n   - 黄色(0.4-0.7)：存在风险因素，需关注\n   - 红色(<0.4)：高风险，需立即干预\n\n3. **生成建议**：基于具体数据给出可操作的个性化干预建议\n\n重要原则：\n- 必须紧扣数据推理，每步引用具体指标\n- 不要编造数据中没有的症状\n- 宁保守勿激进，不确定时倾向低风险\n- 建议要具体可行，不要空泛的"多关心"\n- 如果数据显示正常，就判定为绿色，不强行制造问题"""

    user_prompt = f"""请根据以下学生情绪监测数据，完成心理健康评估。

## 学生信息
- 学生ID: {student_id}
- 监测周期: 过去 {analysis_window_days} 天
- 有效记录数: {record_count}
- 历史情绪基线: {baseline:.2f}

## 数学计算得出的事实数据（不可争议的客观统计）
- 平均情绪分数: {avg_emotion:.3f}（满分1.0）
- 情绪方差: {variance:.4f}
- Shannon 熵值: {entropy:.3f}（0=完全稳定，1=极度紊乱）
- 趋势斜率: {trend_slope:.4f}（正=改善，负=恶化）
- 趋势方向: {trend}
- 情绪突变次数: {abrupt_count} 次
- 恢复次数: {recovery_count} 次
- 平均恢复间隔: {avg_recovery_interval}
- 恢复速度: {recovery_speed:.3f}
- 积极情绪占比: {positive_ratio:.3f}
- 负面情绪占比: {negative_ratio:.3f}
- 情绪分布: {emotion_text}
- Valence 序列（最近）: [{valence_summary}]
- Arousal 序列（最近）: [{arousal_summary}]

## 任务

请基于以上事实数据，运用专业心理学知识进行推理。以严格 JSON 格式返回：

{{
  "indicators": {{
    "emotional_stability_index": 0.0-1.0,
    "negative_emotion_accumulation": 0.0-1.0,
    "social_interaction_frequency": 0.0-1.0,
    "arousal_abnormality_index": 0.0-1.0,
    "sleep_quality_prediction": 0.0-1.0,
    "stress_accumulation_index": 0.0-1.0,
    "overall_mental_health_score": 0.0-1.0
  }},
  "risk_assessment": {{
    "level": "green|yellow|red",
    "reason": "风险判定依据，引用具体数据",
    "reasoning_chain": [
      "第1步: 根据【具体指标】，观察到【具体数值】，这意味着...",
      "第2步: 结合【其他指标】，两者共同指向...",
      "第3步: 综合判断..."
    ]
  }},
  "suggestions": [
    {{
      "priority": "high|medium|low",
      "category": "干预类别",
      "content": "具体可操作建议",
      "target": "班主任|心理老师|家长|学生本人"
    }}
  ],
  "risk_factors": ["风险因素1", "风险因素2"],
  "protective_factors": ["保护因素1", "保护因素2"],
  "detected_patterns": [
    {{
      "pattern_name": "如'学业压力型波动'",
      "confidence": 0.0-1.0,
      "evidence": "数据依据"
    }}
  ],
  "forecast": {{
    "next_period_trend": "改善|稳定|恶化",
    "confidence": 0.0-1.0,
    "key_uncertainty": "最不确定的因素"
  }}
}}

只返回JSON，不要包含其他文字。"""

    try:
        response = client.chat.completions.create(
            model=model or settings.zhipu_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("智谱AI返回空内容")

        result = json.loads(content)
        return result

    except json.JSONDecodeError:
        # JSON解析失败时返回 None，由调用方降级处理
        raise RuntimeError("智谱AI返回内容无法解析为JSON")

    except Exception as e:
        raise RuntimeError(f"智谱AI心理健康分析失败: {str(e)}") from e


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
