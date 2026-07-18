# 反馈模块 API 接入说明

## 概述

反馈模块（多渠道反馈）是心理监测智能体外环工作流的最后一步，负责根据学生心理健康分析结果，通过看板、APP、微信、短信、邮件等多渠道发送分级反馈通知。

v3.0 版本起，反馈模块接入**智谱AI (GLM-4-Flash)**，由AI根据学生的实际分析数据智能生成个性化通知文案，替代原来的固定模板。

---

## API 配置

### 智谱AI (ZhipuAI / BigModel)

| 配置项 | 值 | 位置 |
|--------|-----|------|
| API Key | `d36c6e6b755b453f8b9f52f6541c1136.Pr8uW3uxCXsc923J` | `backend/config.py` |
| Base URL | `https://open.bigmodel.cn/api/paas/v4` | `backend/config.py` |
| 模型 | `glm-4-flash` | `backend/config.py` |

### 配置修改方式

**方式一：直接修改配置文件**

编辑 `backend/config.py`：

```python
class Settings(BaseSettings):
    zhipu_api_key: str = "你的API Key"
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_model: str = "glm-4-flash"
```

**方式二：使用 .env 文件**

在项目根目录创建 `.env` 文件：

```
ZHIPU_API_KEY=你的API Key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL=glm-4-flash
```

---

## 架构说明

```
外环工作流
    │
    ├── aggregate   → 拉取7天情绪数据
    ├── analyze     → 计算12项心理健康指标
    ├── alert       → 生成分级预警（绿/黄/红）
    └── feedback    → 多渠道反馈（本模块）
                        │
                        ▼
              MultiChannelFeedbackTool
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
     智谱AI 生成通知           模板回退（AI不可用时）
     (GLM-4-Flash)              (FALLBACK_TEMPLATES)
            │                       │
            └───────────┬───────────┘
                        ▼
                 分渠道发送通知
              ├─ 看板（Dashboard）
              ├─ APP推送
              ├─ 微信(班主任)
              ├─ 微信(家长)    [仅黄/红]
              ├─ 短信(家长)    [仅红]
              ├─ 邮件(心理教师) [仅红]
              └─ 紧急电话      [仅红]
                        │
                        ▼
             红色预警 → 创建紧急干预工单
```

---

## 核心模块

### 1. 智谱AI客户端 — `backend/llm/zhipu.py`

#### `generate_feedback_content()`

调用智谱AI生成个性化多渠道通知内容。

```python
from backend.llm.zhipu import generate_feedback_content

result = generate_feedback_content(
    student_name="张三",
    severity="yellow",           # green / yellow / red
    risk_reason="近期情绪波动较大",
    overall_score=0.52,
    risk_factors=["负面情绪占比偏高", "情绪呈下降趋势"],
    suggestions=[{"content": "建议班主任关注"}, {"content": "鼓励参加集体活动"}],
    emotion_distribution={"开心": 3, "焦虑": 5, "悲伤": 2, "平静": 4},
    class_name="计算机科学2024",
    student_code="2024-CS-001",
)

# 返回结果
{
    "title": "学生情绪需关注通知",
    "dashboard": "面向心理教师的看板通知内容...",
    "app_push": "面向学生的APP推送内容...",
    "wechat_teacher": "面向班主任的微信通知...",
    "wechat_parent": "面向家长的微信通知...",
    "sms_parent": "面向家长的短信通知...",
    "email_psychologist": "面向心理教师的详细邮件...",
}
```

#### `generate_simple_notification()`

快速生成简短通知（用于非紧急场景）。

```python
from backend.llm.zhipu import generate_simple_notification

text = generate_simple_notification(
    student_name="张三",
    severity="green",
    content="今日情绪状态稳定",
)
```

### 2. 多渠道反馈工具 — `backend/tools/feedback.py`

```python
from backend.tools.feedback import MultiChannelFeedbackTool

tool = MultiChannelFeedbackTool(use_ai=True)  # use_ai=False 可跳过AI使用固定模板

result = tool.execute(
    alert_id=101,
    severity="yellow",
    student_name="张三",
    content="预警原始内容",
    student_info={"class_name": "计算机科学2024", "student_code": "2024-CS-001"},
    analysis_result={
        "risk_reason": "情绪波动",
        "overall_score": 0.52,
        "risk_factors": ["负面情绪占比偏高"],
        "suggestions": [{"content": "建议关注"}],
        "emotion_distribution": {"开心": 3, "焦虑": 5},
    },
)
```

---

## 风险等级与渠道策略

| 等级 | 触发条件 | 通知渠道 | 通知语气 |
|------|---------|---------|---------|
| 🟢 绿色 | 综合评分 ≥ 0.7 | 看板、APP | 鼓励、肯定 |
| 🟡 黄色 | 综合评分 0.4~0.7 | 看板、APP、微信(班主任) | 关切、提醒 |
| 🔴 红色 | 综合评分 < 0.4 | 看板、APP、微信(班主任)、微信(家长)、短信、邮件、紧急电话 | 紧急但镇定、专业 |

---

## 测试验证

### 测试AI连通性

```bash
cd "c:/Users/王建豪/Desktop/大创/大创项目代码"
python -c "
from backend.llm.zhipu import generate_feedback_content
result = generate_feedback_content(
    student_name='测试', severity='green', risk_reason='测试',
    overall_score=0.8, risk_factors=[], suggestions=[], emotion_distribution={}
)
print('AI调用成功！')
print(result['title'])
"
```

### 运行项目测试

```bash
# Prompt解析器测试（12项）
python -m pytest tests/test_prompt.py -v

# 双环状态机测试（30项）
python tests/test_state_machine.py
```

---

## 常见问题

**Q: AI生成的文案不满意怎么办？**

A: 编辑 `backend/llm/zhipu.py` 中 `generate_feedback_content()` 函数的 `system_prompt`，调整AI生成风格。

**Q: 如何关闭AI使用固定模板？**

A: 在代码中将 `MultiChannelFeedbackTool(use_ai=True)` 改为 `use_ai=False`，或当AI调用失败时会自动回退到 `FALLBACK_TEMPLATES`。

**Q: 如何更换为其他大模型？**

A: 智谱AI使用 OpenAI 兼容接口，修改 `backend/config.py` 中的 `zhipu_base_url` 和 `zhipu_model` 即可。如要换其他厂商的API，只需修改 `backend/llm/zhipu.py` 中的客户端初始化。

**Q: API调用失败怎么办？**

A: 模块内置了容错机制：AI调用失败时自动回退到本地模板，不会阻断反馈流程。可以通过修改 `backend/llm/zhipu.py` 中的 `try-except` 块调整重试策略。
