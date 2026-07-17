from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.database import engine, Base, SessionLocal
from backend.models import Student


def seed_data():
    import random
    from datetime import datetime, timedelta
    from backend.models.emotion_record import EmotionRecord

    db = SessionLocal()
    try:
        if db.query(Student).count() == 0:
            db.add_all([
                Student(name="张三", class_name="计算机科学2024", student_code="2024-CS-001", baseline_mood=0.72),
                Student(name="李四", class_name="计算机科学2024", student_code="2024-CS-002", baseline_mood=0.55),
                Student(name="王五", class_name="计算机科学2024", student_code="2024-CS-003", baseline_mood=0.81),
            ])
            db.commit()

        if db.query(EmotionRecord).count() == 0:
            students = db.query(Student).all()
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            EMOTION_VA = {
                "开心": (0.8, 0.6), "平静": (0.3, -0.3), "中性": (0.0, 0.0),
                "焦虑": (-0.4, 0.7), "悲伤": (-0.7, -0.2), "愤怒": (-0.6, 0.8),
                "惊讶": (0.2, 0.9),
            }
            POSITIVE = ["开心", "平静", "中性"]
            NEGATIVE = ["焦虑", "悲伤"]
            OTHER = ["愤怒", "惊讶"]

            records_batch = []
            for day_offset in range(6, 0, -1):
                day = today - timedelta(days=day_offset)
                for student in students:
                    baseline = student.baseline_mood
                    for hour in range(8, 21):
                        minute = random.randint(0, 59)
                        ts = f"{day.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:00"

                        r = random.random()
                        if r < baseline * 0.75:
                            emotion = random.choice(POSITIVE)
                        elif r < baseline * 0.75 + (1 - baseline) * 0.55:
                            emotion = random.choice(NEGATIVE)
                        else:
                            emotion = random.choice(OTHER)

                        va = EMOTION_VA[emotion]
                        valence = va[0] + random.uniform(-0.15, 0.15)
                        arousal = va[1] + random.uniform(-0.15, 0.15)
                        conf = round(0.70 + random.uniform(0, 0.25), 3)
                        score = round(max(0, min(1, 0.5 + valence * 0.25 + arousal * 0.25 + random.uniform(-0.05, 0.05))), 3)

                        records_batch.append(EmotionRecord(
                            student_id=student.id,
                            image_path=f"seed_day{day_offset}_s{student.id}_{hour:02d}{minute:02d}.mp4",
                            facial_emotion=emotion,
                            facial_conf=conf,
                            facial_valence=round(valence, 3),
                            facial_arousal=round(arousal, 3),
                            vestibular_valence=round(valence + random.uniform(-0.1, 0.1), 3),
                            vestibular_arousal=round(arousal + random.uniform(-0.1, 0.1), 3),
                            vestibular_confidence=round(conf - random.uniform(0, 0.1), 3),
                            vestibular_intensity=round(random.uniform(0.3, 0.9), 3),
                            fused_emotion=emotion,
                            fused_score=score,
                            fused_valence=round(valence, 3),
                            fused_arousal=round(arousal, 3),
                            confidence_diff=round(random.uniform(0.02, 0.15), 3),
                            requires_review=0,
                            estimated_accuracy=round(0.90 + random.uniform(0, 0.07), 2),
                            is_manual=0,
                            recorded_at=ts,
                        ))

            db.add_all(records_batch)
            db.commit()
            print(f"  预种 {len(records_batch)} 条历史情绪记录（6天 × 3学生）")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.camera_dir, exist_ok=True)
    os.makedirs("data/obs", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    seed_data()
    from backend.scheduler.jobs import start_scheduler
    start_scheduler()
    yield


app = FastAPI(title="心理监测智能体", version="1.0.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api.routes.upload import router as upload_router
from backend.api.routes.sse import router as sse_router
from backend.api.routes.dashboard import router as dashboard_router
from backend.api.routes.students import router as students_router
from backend.api.routes.alerts import router as alerts_router

app.include_router(upload_router, prefix="/api")
app.include_router(sse_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(students_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "心理监测智能体"}
