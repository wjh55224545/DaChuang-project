from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class EmotionRecordCreate(BaseModel):
    student_id: int
    image_path: str
    trigger_type: str = "manual"


class EmotionRecordOut(BaseModel):
    id: int
    student_id: int
    image_path: str
    facial_emotion: str
    facial_conf: float
    vestibular_valence: float
    vestibular_arousal: float
    fused_emotion: str
    fused_score: float
    is_manual: int
    recorded_at: str

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    success: bool
    image_path: str
    student_id: int
    trigger_type: str
    run_id: str | None = None


class AgentTriggerResponse(BaseModel):
    success: bool
    run_id: str
    message: str
    records_created: int = 0


class StudentOut(BaseModel):
    id: int
    name: str
    class_name: str
    student_code: str
    baseline_mood: float
    created_at: str

    model_config = {"from_attributes": True}


class StudentDetailOut(StudentOut):
    recent_records: list[EmotionRecordOut] = []
    recent_alerts: list["AlertOut"] = []


SSE_EVENT_THOUGHT = "thought"
SSE_EVENT_ACTION = "action"
SSE_EVENT_OBSERVATION = "observation"
SSE_EVENT_FINAL = "final"
SSE_EVENT_ERROR = "error"
