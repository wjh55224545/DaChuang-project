from __future__ import annotations
import json
import logging
from backend.database import SessionLocal
from backend.models.student import Student
from backend.models.alert import Alert
from backend.tools.feedback import MultiChannelFeedbackTool
from backend.graph.state import OuterLoopState

logger = logging.getLogger(__name__)


def feedback_node(state: OuterLoopState) -> dict:
    """
    反馈节点 - 执行多渠道预警通知

    增强版功能：
    - 根据风险等级动态选择反馈渠道
    - 绿色→APP；黄色→APP+班主任；红色→APP+班主任+心理教师+家长
    - 红色状态触发紧急干预工单
    - 接入智谱AI智能生成个性化通知内容
    """
    alerts_generated = state.get("alerts_generated", [])
    analysis_results = state.get("analysis_results", {})

    if not alerts_generated:
        logger.info("反馈节点：无预警需要处理")
        return {"feedback_sent": {}}

    tool = MultiChannelFeedbackTool(use_ai=True)
    feedback_sent: dict[int, dict] = {}

    db = SessionLocal()
    try:
        for alert in alerts_generated:
            try:
                student_id = alert.get("student_id")
                if student_id is not None:
                    student_id = int(student_id)

                severity = alert.get("severity", "green")
                alert_id = int(alert.get("alert_id", 0))

                # 获取学生详细信息
                student = db.query(Student).filter(Student.id == student_id).first() if student_id else None

                student_info = {
                    "student_id": student_id,
                    "student_code": student.student_code if student else "unknown",
                    "name": student.name if student else alert.get("student_name", "未知"),
                    "class_name": student.class_name if student else "",
                    "school": getattr(student, 'school', ''),
                    "baseline_mood": student.baseline_mood if student else 0.7,
                }

                # 获取分析结果
                analysis = {}
                if student_id is not None:
                    analysis = analysis_results.get(student_id) or analysis_results.get(str(student_id)) or {}

                # 执行多渠道反馈（AI生成通知内容）
                result = tool.execute(
                    alert_id=alert_id,
                    severity=str(severity),
                    student_name=str(student_info["name"]),
                    content=str(alert.get("content", "")),
                    student_info=student_info,
                    analysis_result=analysis,
                )

                feedback_sent[alert_id] = result.data if result.success else {"error": result.error}

                # ---- 将AI生成的内容回写到 Alert 数据库记录 ----
                if result.success:
                    notif = result.data.get("notification_content", {})
                    ai_generated = result.data.get("ai_generated", False)

                    # 在终端打印AI生成结果
                    print(f"\n{'='*50}")
                    print(f"🤖 智谱AI生成通知 | 学生: {student_info['name']} | 等级: {severity} | 评分: {analysis.get('overall_score', 'N/A')}")
                    print(f"{'='*50}")
                    channel_labels = {
                        "dashboard": "看板", "app_push": "APP推送",
                        "wechat_teacher": "微信(班主任)", "wechat_parent": "微信(家长)",
                        "sms_parent": "短信(家长)", "email_psychologist": "邮件(心理教师)",
                    }
                    for ch_key, ch_label in channel_labels.items():
                        text = notif.get(ch_key, "")
                        if text:
                            print(f"  [{ch_label}] {text[:100]}{'...' if len(text)>100 else ''}")

                    if result.data.get("work_order"):
                        wo = result.data["work_order"]
                        print(f"  🚨 紧急工单: {wo['work_order_id']} | 优先级: {wo['priority']} | 截止: {wo['due_time']}")
                    print(f"  {'✅ AI生成' if ai_generated else '⚠️ 模板回退'}")
                    print(f"{'='*50}\n")

                    # 回写数据库：更新 Alert 的反馈内容为AI生成的内容
                    alert_record = db.query(Alert).filter(Alert.id == alert_id).first()
                    if alert_record:
                        # 把完整的AI通知内容写入 feedback_content 字段
                        alert_record.feedback_content = json.dumps({
                            "title": notif.get("title", ""),
                            "dashboard": notif.get("dashboard", ""),
                            "app_push": notif.get("app_push", ""),
                            "wechat_teacher": notif.get("wechat_teacher", ""),
                            "wechat_parent": notif.get("wechat_parent", ""),
                            "sms_parent": notif.get("sms_parent", ""),
                            "email_psychologist": notif.get("email_psychologist", ""),
                            "ai_generated": ai_generated,
                        }, ensure_ascii=False)
                        alert_record.sent_channels = json.dumps(
                            result.data.get("sent_channels", []), ensure_ascii=False
                        )

            except Exception as e:
                logger.exception("反馈节点处理预警失败: alert_id=%s, error=%s", alert.get("alert_id"), e)
                feedback_sent[int(alert.get("alert_id", 0))] = {"error": str(e)}

        db.commit()
        return {"feedback_sent": feedback_sent}

    finally:
        db.close()
