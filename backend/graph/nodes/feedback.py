from __future__ import annotations
import logging
from backend.database import SessionLocal
from backend.models.student import Student
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
                # 统一student_id类型
                student_id = alert.get("student_id")
                if student_id is not None:
                    student_id = int(student_id)

                severity = alert.get("severity", "green")

                # 获取学生详细信息
                student = db.query(Student).filter(Student.id == student_id).first() if student_id else None

                # 构建学生信息
                student_info = {
                    "student_id": student_id,
                    "student_code": student.student_code if student else "unknown",
                    "name": student.name if student else alert.get("student_name", "未知"),
                    "class_name": student.class_name if student else "",
                    "school": getattr(student, 'school', ''),
                    "baseline_mood": student.baseline_mood if student else 0.7,
                }

                # 获取分析结果（统一key类型）
                analysis = {}
                if student_id is not None:
                    # 尝试用int key和str key查找
                    analysis = analysis_results.get(student_id) or analysis_results.get(str(student_id)) or {}

                # 执行多渠道反馈
                result = tool.execute(
                    alert_id=int(alert.get("alert_id", 0)),
                    severity=str(severity),
                    student_name=str(student_info["name"]),
                    content=str(alert.get("content", "")),
                    student_info=student_info,
                    analysis_result=analysis,
                )

                alert_key = int(alert.get("alert_id", 0))
                feedback_sent[alert_key] = result.data if result.success else {"error": result.error}

            except Exception as e:
                logger.exception("反馈节点处理预警失败: alert_id=%s, error=%s", alert.get("alert_id"), e)
                alert_key = int(alert.get("alert_id", 0))
                feedback_sent[alert_key] = {"error": str(e)}

        return {"feedback_sent": feedback_sent}

    finally:
        db.close()
