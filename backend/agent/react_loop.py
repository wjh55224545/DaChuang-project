from __future__ import annotations
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool as langchain_tool

from backend.llm.deepseek import get_llm
from backend.tools.emotion_recognition import EmotionRecognitionTool
from backend.tools.obs_storage import OBSPersistenceTool
from backend.tools.mental_health import MentalHealthAnalysisTool
from backend.tools.feedback import MultiChannelFeedbackTool


def create_analysis_agent():
    """创建用于外环analyze节点的ReAct agent。"""
    emo = EmotionRecognitionTool()
    obs = OBSPersistenceTool()
    mh = MentalHealthAnalysisTool()
    fb = MultiChannelFeedbackTool()

    @langchain_tool
    def emotion_recognition(image_path: str, student_id: int = 0) -> dict:
        """多模态情绪识别：分析图片中人物的情绪状态，融合面部微表情和振动感知数据。"""
        result = emo.execute(image_path=image_path, student_id=student_id)
        return result.data if result.success else {"error": result.error}

    @langchain_tool
    def obs_persistence(record_id: int, data: dict) -> dict:
        """华为云OBS持久化：将情绪记录存入云端存储。"""
        result = obs.execute(record_id=record_id, data=data)
        return result.data if result.success else {"error": result.error}

    @langchain_tool
    def mental_health_analysis(student_id: int, records: list, baseline: float = 0.7) -> dict:
        """时序心理分析：分析学生情绪时序数据，检测异常模式和风险趋势。"""
        result = mh.execute(student_id=student_id, records=records, baseline=baseline)
        return result.data if result.success else {"error": result.error}

    @langchain_tool
    def multi_channel_feedback(alert_id: int, severity: str, student_name: str, content: str) -> dict:
        """多渠道反馈：根据预警等级通过看板/微信/邮件/短信发送通知。"""
        result = fb.execute(alert_id=alert_id, severity=severity, student_name=student_name, content=content)
        return result.data if result.success else {"error": result.error}

    llm = get_llm(streaming=True)
    tools = [emotion_recognition, obs_persistence, mental_health_analysis, multi_channel_feedback]

    from prompts.system_prompt import SYSTEM_PROMPT

    return create_react_agent(model=llm, tools=tools, state_modifier=SYSTEM_PROMPT)
