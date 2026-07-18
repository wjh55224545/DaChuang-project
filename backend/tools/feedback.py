"""
多渠道反馈工具 (MultiChannelFeedbackTool)
==========================================

功能说明：
- 根据风险等级动态选择反馈渠道
- 绿色状态：推送至学生个人APP
- 黄色状态：同时通知学生与班主任
- 红色状态：触发紧急流程，自动通知心理教师与家长

通知渠道：
- 看板（Dashboard）：系统预警面板展示
- APP（移动应用）：学生端APP推送
- 微信（WeChat）：企业微信/公众号模板消息
- 短信（SMS）：运营商短信通道
- 邮件（Email）：学校邮箱系统
- 紧急电话（Emergency Call）：自动外呼

触发流程：
绿色 → 看板 + APP
黄色 → 看板 + APP + 微信(班主任)
红色 → 看板 + APP + 微信(班主任) + 微信(家长) + 短信 + 邮件 + 紧急工单

v3.0 更新：接入智谱AI (GLM-4) 智能生成个性化通知内容，替代原有固定模板。
"""

from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List

from backend.tools.base import BaseTool, ToolResult
from backend.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 渠道配置
# ---------------------------------------------------------------------------

CHANNELS = {
    "dashboard": {
        "name": "看板",
        "enabled": True,
        "delay": 0,
        "priority": "low",
    },
    "app": {
        "name": "学生APP",
        "enabled": True,
        "delay": 1,
        "priority": "medium",
    },
    "wechat_teacher": {
        "name": "微信(班主任)",
        "enabled": True,
        "delay": 2,
        "priority": "medium",
    },
    "wechat_parent": {
        "name": "微信(家长)",
        "enabled": True,
        "delay": 2,
        "priority": "high",
    },
    "sms_parent": {
        "name": "短信(家长)",
        "enabled": True,
        "delay": 3,
        "priority": "high",
    },
    "email_psychologist": {
        "name": "邮件(心理教师)",
        "enabled": True,
        "delay": 2,
        "priority": "high",
    },
    "phone_emergency": {
        "name": "紧急电话",
        "enabled": True,
        "delay": 5,
        "priority": "critical",
    },
}

# ---------------------------------------------------------------------------
# 风险等级对应的渠道策略
# ---------------------------------------------------------------------------

CHANNEL_STRATEGY = {
    "green": ["dashboard", "app"],
    "yellow": ["dashboard", "app", "wechat_teacher"],
    "red": [
        "dashboard", "app", "wechat_teacher",
        "wechat_parent", "sms_parent",
        "email_psychologist", "phone_emergency",
    ],
}

# ---------------------------------------------------------------------------
# 后备通知模板（当智谱AI不可用时使用）
# ---------------------------------------------------------------------------

FALLBACK_TEMPLATES = {
    "green": {
        "title": "🌿 心理健康状态良好",
        "content": "{student_name}同学今日情绪状态稳定，继续保持哦！",
        "dashboard": "学生 {student_name} 今日心理健康状态良好，综合评分 {score}。",
        "app_push": "{student_name}同学，今日情绪状态良好，继续保持哦！",
        "wechat_teacher": "",
        "wechat_parent": "",
        "sms_parent": "",
        "email_psychologist": "",
    },
    "yellow": {
        "title": "⚠️ 需要关注",
        "content": "{student_name}同学近期情绪有些波动，建议多加留意。详情请查看系统。",
        "dashboard": "【关注】学生 {student_name} 心理健康状态需要关注。风险因素：{risk_reasons}。建议：{suggestions}。",
        "app_push": "{student_name}同学，近期有些情绪波动，记得好好照顾自己哦～",
        "wechat_teacher": "【心理健康预警】{student_name} 同学（班级：{class_name}）近期情绪状态需要关注。请班主任留意并适当关怀。建议：{suggestions}",
        "wechat_parent": "",
        "sms_parent": "",
        "email_psychologist": "",
    },
    "red": {
        "title": "🚨 紧急预警",
        "content": "{student_name}同学情绪状态异常，需要立即关注！",
        "dashboard": "【紧急】学生 {student_name} 心理健康状态异常！综合评分 {score}。风险因素：{risk_reasons}。需立即启动干预流程。",
        "app_push": "{student_name}同学，我们注意到你最近可能遇到了一些困难，学校的心理老师随时愿意倾听和支持你。",
        "wechat_teacher": "【紧急预警】{student_name} 同学情绪状态异常（班级：{class_name}），请立即联系学生并了解情况。风险因素：{risk_reasons}。建议：{suggestions}",
        "wechat_parent": "【心理健康预警】{student_name} 家长您好，您的孩子近期情绪状态需要关注。学校心理教师将尽快与您联系了解情况。如有紧急情况，请拨打学校心理热线。",
        "sms_parent": "【心镜智能】{student_name}家长，您的孩子近期情绪状态需关注。学校心理教师将尽快与您联系。",
        "email_psychologist": "【紧急工单】学生 {student_name}（学号：{student_code}，班级：{class_name}）心理健康状态异常，需安排一对一访谈。综合评分：{score}。风险因素：{risk_reasons}。建议：{suggestions}",
    },
}


class MultiChannelFeedbackTool(BaseTool):
    """
    多渠道反馈工具

    v3.0 更新：
    - 接入智谱AI (GLM-4) 智能生成个性化通知内容
    - AI不可用时自动回退到预设模板
    - 保持渠道策略和工单系统逻辑不变
    """

    name = "多渠道反馈"
    description = (
        "根据预警等级通过看板、APP、微信、邮件、短信等多渠道发送分级反馈通知。"
        "绿色→APP；黄色→APP+班主任；红色→APP+班主任+心理教师+家长，触发紧急干预流程。"
        "接入智谱AI智能生成个性化通知内容。"
    )

    def __init__(self, use_ai: bool = True):
        """
        Args:
            use_ai: 是否使用智谱AI生成通知内容。False时使用固定模板。
        """
        super().__init__()
        self._send_count = 0
        self._use_ai = use_ai
        self._settings = get_settings()

    def execute(
        self,
        alert_id: int = 0,
        severity: str = "green",
        student_name: str = "",
        content: str = "",
        student_info: dict | None = None,
        analysis_result: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        """
        执行多渠道反馈

        Args:
            alert_id: 预警ID
            severity: 风险等级 (green/yellow/red)
            student_name: 学生姓名
            content: 预警原始内容
            student_info: 学生详细信息 {student_code, class_name, school}
            analysis_result: 分析结果数据（含12项指标、风险因素、建议等）
        """
        self._send_count += 1

        try:
            info = student_info or {}
            analysis = analysis_result or {}

            # 步骤1: 确定发送渠道
            channels_to_send = CHANNEL_STRATEGY.get(severity, CHANNEL_STRATEGY["green"])

            # 步骤2: 智能生成通知内容（优先使用AI，失败时回退到模板）
            notification_content = self._generate_notification_content(
                student_name=student_name,
                severity=severity,
                content=content,
                student_info=info,
                analysis_result=analysis,
            )

            # 步骤3: 分渠道发送
            send_results = {}
            for channel in channels_to_send:
                channel_content = self._get_channel_content(channel, notification_content, severity)
                result = self._send_to_channel(
                    channel=channel,
                    alert_id=alert_id,
                    content=channel_content,
                    severity=severity,
                    student_name=student_name,
                )
                send_results[channel] = result

            # 步骤4: 生成紧急干预工单（红色级别）
            work_order = None
            if severity == "red":
                work_order = self._create_emergency_work_order(
                    alert_id=alert_id,
                    student_name=student_name,
                    student_info=info,
                    analysis_result=analysis,
                    send_results=send_results,
                )

            # 统计发送结果
            success_count = sum(1 for r in send_results.values() if r["success"])
            total_count = len(send_results)

            return ToolResult(
                success=True,
                data={
                    # 发送状态摘要
                    "sent_channels": [CHANNELS[c]["name"] for c in channels_to_send],
                    "channel_count": total_count,
                    "success_count": success_count,
                    "failure_count": total_count - success_count,
                    "success_rate": round(success_count / total_count, 2) if total_count > 0 else 0,

                    # 各渠道详细结果
                    "channel_results": send_results,

                    # 预警工单（红色级别）
                    "work_order": work_order,

                    # 通知内容（AI生成或模板）
                    "notification_content": notification_content,

                    # 元数据
                    "alert_id": alert_id,
                    "severity": severity,
                    "student_name": student_name,
                    "send_timestamp": datetime.now().isoformat(),
                    "message_id": f"msg-{hashlib.md5(f'{alert_id}{self._send_count}'.encode()).hexdigest()[:12]}",
                    "ai_generated": notification_content.get("ai_generated", False),
                },
            )

        except Exception as e:
            logger.exception("多渠道反馈发送失败: %s", e)
            return ToolResult(
                success=False,
                data={},
                error=f"多渠道反馈发送失败: {str(e)}",
            )

    # ------------------------------------------------------------------
    # 通知内容生成
    # ------------------------------------------------------------------

    def _generate_notification_content(
        self,
        student_name: str,
        severity: str,
        content: str,
        student_info: dict,
        analysis_result: dict,
    ) -> dict:
        """
        生成通知内容：优先使用智谱AI，失败则回退到本地模板。
        """
        if self._use_ai:
            try:
                return self._generate_with_ai(
                    student_name=student_name,
                    severity=severity,
                    content=content,
                    student_info=student_info,
                    analysis_result=analysis_result,
                )
            except Exception as e:
                logger.warning("智谱AI通知生成失败，回退到本地模板: %s", e)

        return self._generate_with_template(
            student_name=student_name,
            severity=severity,
            content=content,
            student_info=student_info,
            analysis_result=analysis_result,
        )

    def _generate_with_ai(
        self,
        student_name: str,
        severity: str,
        content: str,
        student_info: dict,
        analysis_result: dict,
    ) -> dict:
        """
        调用智谱AI生成个性化通知内容。
        """
        from backend.llm.zhipu import generate_feedback_content

        # 提取AI生成所需参数
        risk_reason = analysis_result.get("risk_reason", content)
        overall_score = analysis_result.get("overall_score", 0.5)
        risk_factors = analysis_result.get("risk_factors", [])
        suggestions = analysis_result.get("suggestions", [])
        emotion_distribution = analysis_result.get("emotion_distribution", {})

        ai_result = generate_feedback_content(
            student_name=student_name,
            severity=severity,
            risk_reason=str(risk_reason),
            overall_score=float(overall_score),
            risk_factors=risk_factors,
            suggestions=suggestions,
            emotion_distribution=emotion_distribution,
            class_name=student_info.get("class_name", ""),
            student_code=student_info.get("student_code", ""),
        )

        # 将AI生成结果与格式化数据合并
        ai_result["formatted_data"] = {
            "student_name": student_name,
            "class_name": student_info.get("class_name", "未知班级"),
            "student_code": student_info.get("student_code", "未知学号"),
            "school": student_info.get("school", ""),
            "score": str(overall_score),
        }
        ai_result["ai_generated"] = True
        return ai_result

    def _generate_with_template(
        self,
        student_name: str,
        severity: str,
        content: str,
        student_info: dict,
        analysis_result: dict,
    ) -> dict:
        """
        使用本地模板生成通知内容（原有逻辑，作为后备方案）。
        """
        templates = FALLBACK_TEMPLATES.get(severity, FALLBACK_TEMPLATES["green"])

        # 提取风险因素和建议
        risk_factors = analysis_result.get("risk_factors", [])
        risk_reasons = "；".join(risk_factors) if risk_factors else content

        suggestions = analysis_result.get("suggestions", [])
        if suggestions and isinstance(suggestions[0], dict):
            suggestion_texts = [s.get("content", "") for s in suggestions[:2]]
            suggestion_text = "；".join(filter(None, suggestion_texts))
        else:
            suggestion_text = str(suggestions[0]) if suggestions else "建议持续关注"

        # 构建格式化数据
        format_data = {
            "student_name": student_name,
            "class_name": student_info.get("class_name", "未知班级"),
            "student_code": student_info.get("student_code", "未知学号"),
            "school": student_info.get("school", ""),
            "score": str(analysis_result.get("overall_score", analysis_result.get("attention_score", 0))),
            "risk_reasons": risk_reasons,
            "suggestions": suggestion_text or "建议持续关注",
            "content": content,
        }

        # 使用模板格式化各渠道内容
        contents = {}
        for key, template in templates.items():
            if isinstance(template, str) and template:
                try:
                    contents[key] = template.format(**format_data)
                except KeyError:
                    contents[key] = template

        return {
            "title": contents.get("title", templates.get("title", "")),
            "content": contents.get("content", ""),
            "dashboard": contents.get("dashboard", ""),
            "app_push": contents.get("app_push", content),
            "wechat_teacher": contents.get("wechat_teacher", ""),
            "wechat_parent": contents.get("wechat_parent", ""),
            "sms_parent": contents.get("sms_parent", ""),
            "email_psychologist": contents.get("email_psychologist", ""),
            "formatted_data": format_data,
            "ai_generated": False,
        }

    # ------------------------------------------------------------------
    # 渠道发送
    # ------------------------------------------------------------------

    def _get_channel_content(
        self,
        channel: str,
        notification_content: dict,
        severity: str,
    ) -> str:
        """
        从通知内容中提取对应渠道的文本。
        """
        # 渠道到内容字段的映射
        channel_content_map = {
            "dashboard": "dashboard",
            "app": "app_push",
            "wechat_teacher": "wechat_teacher",
            "wechat_parent": "wechat_parent",
            "sms_parent": "sms_parent",
            "email_psychologist": "email_psychologist",
            "phone_emergency": "dashboard",  # 紧急电话使用看板摘要作为话术参考
        }

        content_key = channel_content_map.get(channel, "dashboard")
        channel_text = notification_content.get(content_key, "")

        # 如果没有该渠道的特定内容，使用通用内容
        if not channel_text:
            channel_text = notification_content.get("content", "")

        return channel_text

    def _send_to_channel(
        self,
        channel: str,
        alert_id: int,
        content: str,
        severity: str,
        student_name: str,
    ) -> dict:
        """
        向指定渠道发送通知。

        当前各渠道发送为模拟实现。真实对接时：
        - app: 调用极光推送/个推 SDK
        - wechat_*: 调用企业微信/公众号模板消息 API
        - sms_parent: 调用阿里云短信/腾讯云短信 API
        - email_psychologist: 调用 SMTP 或 SendGrid API
        - phone_emergency: 调用语音呼叫 API（如阿里云语音通知）
        """
        channel_info = CHANNELS.get(channel, {})
        channel_name = channel_info.get("name", channel)

        # 模拟发送结果（95%成功率）
        import random
        import time

        success = random.random() > 0.05

        result = {
            "success": success,
            "channel": channel,
            "channel_name": channel_name,
            "priority": channel_info.get("priority", "low"),
            "sent_at": datetime.now().isoformat(),
            "message_id": f"{channel}-{hashlib.md5(f'{alert_id}{channel}'.encode()).hexdigest()[:8]}",
            "content_preview": content[:100] if content else "",
        }

        if success:
            result["delivery_status"] = "delivered"
            result["read_status"] = "unread"

            # 渠道特定元数据
            if channel == "app":
                result["device_type"] = random.choice(["iOS", "Android"])
                result["push_id"] = f"push-{random.randint(100000, 999999)}"
            elif channel == "sms_parent":
                result["phone_number"] = "138****8888"
                result["sms_signature"] = "【心镜智能】"
            elif channel == "email_psychologist":
                result["email_address"] = "psych@school.edu.cn"
                result["subject"] = f"[{severity.upper()}] 学生心理预警通知 - {student_name}"
            elif channel == "phone_emergency":
                result["call_status"] = random.choice(["dialing", "answered", "missed"])
                result["call_duration"] = random.randint(0, 120) if result["call_status"] == "answered" else 0
        else:
            result["delivery_status"] = "failed"
            result["error_message"] = "通道暂时不可用"
            result["retry_recommended"] = True

        return result

    # ------------------------------------------------------------------
    # 紧急干预工单
    # ------------------------------------------------------------------

    def _create_emergency_work_order(
        self,
        alert_id: int,
        student_name: str,
        student_info: dict | None,
        analysis_result: dict | None,
        send_results: dict,
    ) -> dict:
        """
        创建紧急干预工单（红色预警触发）。

        真实实现应接入学校工单系统或OA系统API。
        """
        info = student_info or {}
        analysis = analysis_result or {}

        # 生成工单号
        work_order_id = (
            f"WO-{datetime.now().strftime('%Y%m%d')}-"
            f"{hashlib.md5(f'{alert_id}'.encode()).hexdigest()[:6].upper()}"
        )

        # 统计各渠道发送结果
        channel_summary = []
        for ch, result in send_results.items():
            channel_summary.append({
                "channel": CHANNELS.get(ch, {}).get("name", ch),
                "status": result.get("delivery_status", "unknown"),
            })

        return {
            "work_order_id": work_order_id,
            "alert_id": alert_id,
            "student_name": student_name,
            "student_code": info.get("student_code", ""),
            "class_name": info.get("class_name", ""),
            "priority": "P0",
            "status": "pending",
            "assigned_to": "心理教师组",
            "due_time": (datetime.now() + timedelta(hours=24)).isoformat(),
            "description": (
                f"学生{student_name}情绪状态异常，"
                f"综合评分{analysis.get('overall_score', 0):.2f}，需安排一对一访谈。"
            ),
            "risk_level": "red",
            "risk_factors": analysis.get("risk_factors", []),
            "suggestions": [
                s.get("content", "") if isinstance(s, dict) else str(s)
                for s in analysis.get("suggestions", [])
            ],
            "channel_summary": channel_summary,
            "created_at": datetime.now().isoformat(),
            "creator": "心镜智能体",
            "work_order_type": "psychological_intervention",
            "follow_up_required": True,
            "escalation_enabled": True,
        }

    # ------------------------------------------------------------------
    # 通知状态查询
    # ------------------------------------------------------------------

    def query_notification_status(self, message_ids: List[str]) -> ToolResult:
        """
        查询通知送达状态。

        当前为模拟实现。真实对接时需调用对应渠道的回执查询API。
        """
        import random

        status_results = []
        for msg_id in message_ids:
            status_results.append({
                "message_id": msg_id,
                "status": random.choice(["delivered", "read", "failed"]),
                "delivered_at": datetime.now().isoformat() if random.random() > 0.1 else None,
                "read_at": datetime.now().isoformat() if random.random() > 0.5 else None,
            })

        return ToolResult(
            success=True,
            data={
                "query_count": len(message_ids),
                "results": status_results,
                "query_time": datetime.now().isoformat(),
            },
        )
