from __future__ import annotations
from datetime import datetime
from sqlalchemy import String, Float, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class EmotionRecord(Base):
    __tablename__ = "emotion_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(Integer, ForeignKey("students.id"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)

    # 面部微表情识别结果
    facial_emotion: Mapped[str] = mapped_column(String(32), nullable=False)
    facial_conf: Mapped[float] = mapped_column(Float, nullable=False)
    facial_valence: Mapped[float] = mapped_column(Float, nullable=False)
    facial_arousal: Mapped[float] = mapped_column(Float, nullable=False)

    # 前庭振动识别结果
    vestibular_valence: Mapped[float] = mapped_column(Float, nullable=False)
    vestibular_arousal: Mapped[float] = mapped_column(Float, nullable=False)
    vestibular_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    vestibular_intensity: Mapped[float] = mapped_column(Float, nullable=False)

    # 融合结果
    fused_emotion: Mapped[str] = mapped_column(String(32), nullable=False)
    fused_score: Mapped[float] = mapped_column(Float, nullable=False)
    fused_valence: Mapped[float] = mapped_column(Float, nullable=False)
    fused_arousal: Mapped[float] = mapped_column(Float, nullable=False)

    # 质量指标
    confidence_diff: Mapped[float] = mapped_column(Float, nullable=False)
    requires_review: Mapped[int] = mapped_column(Integer, default=0)
    estimated_accuracy: Mapped[float] = mapped_column(Float, nullable=False)

    # 元数据
    is_manual: Mapped[int] = mapped_column(Integer, default=0)
    recorded_at: Mapped[str] = mapped_column(String(32), default=lambda: datetime.now().isoformat())

    student = relationship("Student", back_populates="emotion_records")
