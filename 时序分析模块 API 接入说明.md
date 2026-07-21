# 时序分析模块 API 接入说明

## 概述

时序分析模块（心理健康指标计算）是心理监测智能体外环工作流的核心环节，负责根据学生7天情绪时序数据，计算12项心理健康指标、判定风险等级并生成干预建议。

v4.0 版本起，采用**两层混合架构**：
- **Layer 1（纯数学计算，5项）**：Shannon熵值、线性回归趋势、恢复速度、积极情绪占比、突变检测
- **Layer 2（智谱AI GLM-4-Flash 推理，7项）**：由大模型根据数学事实数据进行专业推理
- **自动降级**：LLM不可用时回退 rule-engine（原 LSTM-Transformer 模拟 + 规则引擎），保证系统可用

同一套智谱AI配置，与反馈模块共用 API Key 和 endpoint。

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
    ├── analyze     → 计算12项心理健康指标（本模块）
    ├── alert       → 生成分级预警（绿/黄/红）
    └── feedback    → 多渠道反馈
                        │
                        ▼
              MentalHealthAnalysisTool
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
    Layer 1: 纯数学计算      Layer 2: 智谱AI推理
    ┌──────────────────┐    ┌──────────────────────┐
    │ ② 情绪波动熵值    │    │ ① 情绪稳定性指数       │
    │ ⑤ 日间情绪趋势    │    │ ③ 负面情绪累积度       │
    │ ⑦ 情绪恢复速度    │    │ ④ 社交互动频次         │
    │ ⑩ 积极情绪占比    │    │ ⑥ 唤醒度异常指数       │
    │ ⑪ 情绪突变检测    │    │ ⑧ 睡眠质量预测         │
    └──────────────────┘    │ ⑨ 压力累积指数         │
             │              │ ⑫ 综合心理健康评分       │
             │              ├─ 风险评估（绿/黄/红）   │
             │              ├─ 推理链（reasoning_chain）│
             │              ├─ 干预建议（suggestions）  │
             │              ├─ 风险因素（risk_factors） │
             │              ├─ 保护因素（protective）   │
             │              ├─ 行为模式识别（patterns） │
             │              └─ 趋势预测（forecast）     │
             │              └──────────────────────┘
             │                        │
             │              LLM 不可用时自动降级
             │                        │
             │              ┌─────────▼──────────┐
             │              │ rule-engine 回退   │
             │              │ (LSTM-Transformer  │
             │              │  模拟 + 规则引擎)  │
             │              └────────────────────┘
             │                        │
             └────────────┬───────────┘
                          ▼
                12项指标完整输出 + 风险等级 + 建议
```

---

## 核心模块

### 1. 智谱AI客户端 — `backend/llm/zhipu.py`

#### `generate_mental_health_analysis()` [新增]

调用智谱AI（GLM-4-Flash），基于数学事实数据推理7项心理健康指标并进行风险评估。

**请求参数：**

```python
from backend.llm.zhipu import generate_mental_health_analysis

result = generate_mental_health_analysis(
    student_id=1001,
    baseline=0.65,                    # 历史情绪基线
    avg_emotion=0.583,                # 平均情绪分数
    variance=0.0231,                  # 情绪方差
    entropy=0.412,                    # Shannon 熵值
    trend_slope=-0.0123,              # 趋势斜率（正=改善，负=恶化）
    trend="稳定",                     # 趋势方向：改善中/稳定/下降中
    abrupt_count=2,                   # 情绪突变次数
    recovery_speed=0.333,             # 恢复速度
    recovery_count=3,                 # 恢复次数
    avg_recovery_interval="9.0h",     # 平均恢复间隔
    positive_ratio=0.400,             # 积极情绪占比
    negative_ratio=0.350,             # 负面情绪占比
    emotion_distribution={"开心": 4, "焦虑": 3, "悲伤": 2, "平静": 1},
    valence_series=[0.6, 0.5, 0.7, 0.4, 0.6, 0.5, 0.8, 0.3, 0.6, 0.7],
    arousal_series=[0.3, 0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.1, 0.4, 0.3],
    record_count=14,                  # 有效记录数
    analysis_window_days=7,           # 分析窗口天数
)
```

**返回结果：**

```json
{
  "indicators": {
    "emotional_stability_index": 0.650,
    "negative_emotion_accumulation": 0.450,
    "social_interaction_frequency": 0.600,
    "arousal_abnormality_index": 0.200,
    "sleep_quality_prediction": 0.720,
    "stress_accumulation_index": 0.420,
    "overall_mental_health_score": 0.620
  },
  "risk_assessment": {
    "level": "yellow",
    "reason": "综合评分0.62偏低，负面情绪累积度0.45需关注",
    "reasoning_chain": [
      "第1步: 根据熵值0.412，情绪存在中等程度波动",
      "第2步: 结合趋势斜率-0.0123，情绪呈轻微下降趋势",
      "第3步: 综合判断为黄色风险等级，建议班主任关注"
    ]
  },
  "suggestions": [
    {
      "priority": "medium",
      "category": "情绪调节",
      "content": "建议每天安排15分钟正念练习",
      "target": "学生本人"
    }
  ],
  "risk_factors": ["负面情绪占比偏高", "情绪呈轻微下降趋势"],
  "protective_factors": ["具备一定的情绪恢复能力", "情绪突变不频繁"],
  "detected_patterns": [
    {
      "pattern_name": "学业压力型波动",
      "confidence": 0.75,
      "evidence": "工作日情绪偏低，周末回升"
    }
  ],
  "forecast": {
    "next_period_trend": "稳定",
    "confidence": 0.80,
    "key_uncertainty": "近期是否有考试或重大事件未知"
  }
}
```

**模型参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| temperature | 0.3 | 低温度保证输出稳定性 |
| max_tokens | 2048 | 足够容纳完整JSON响应 |
| response_format | `{"type": "json_object"}` | 强制JSON输出 |

### 2. 时序分析工具 — `backend/tools/mental_health.py`

#### `_llm_deep_analysis()` [新增]

内部方法，负责组装参数并调用 `generate_mental_health_analysis()`。

```python
# 由 execute() 内部自动调用，无需手动调用
llm_result = self._llm_deep_analysis(
    student_id=1001,
    baseline=0.65,
    indicators={...},          # Layer 1 已计算的指标
    emotion_series=[...],      # 原始情绪时间序列
    analysis_window_days=7,
)
```

#### `_calculate_recovery_details()` [新增]

内部方法，从情绪序列中提取恢复相关统计数据。

```python
recovery_speed, recovery_count, avg_recovery_interval = \
    self._calculate_recovery_details(emotion_series)
# 返回值示例: (0.333, 3, "9.0h")
```

### 3. execute() 主流程

```python
from backend.tools.mental_health import MentalHealthAnalysisTool

tool = MentalHealthAnalysisTool()

result = tool.execute(
    student_id=1001,
    records=[...],             # 当天情绪记录
    baseline=0.65,             # 历史基线
    obs_records=[...],         # OBS拉取的7天历史数据
    analysis_window_days=7,
)

# result.data 包含:
# - indicators: 12项指标（已由LLM覆盖或规则引擎兜底）
# - risk_level / risk_reason: 风险等级及原因
# - lstm_transformer_analysis: 包含 source / reasoning_chain / forecast 等
# - suggestions / risk_factors / protective_factors
# - model_version: "ZhipuGLM-4-Flash-v1.0" 或 "RuleEngine-Fallback-v2.1"
```

---

## 12项指标分层明细

| 编号 | 指标名称 | 计算层 | 计算方式 |
|------|---------|--------|---------|
| ① | 情绪稳定性指数 | Layer 2 (LLM) | 综合熵、方差、突变次数推理 |
| ② | 情绪波动熵值 | Layer 1 (数学) | Shannon熵 + 5区间离散化 |
| ③ | 负面情绪累积度 | Layer 2 (LLM) | 综合负面占比、趋势、分布推理 |
| ④ | 社交互动频次 | Layer 2 (LLM) | 基于记录密度 + 时间分布推理 |
| ⑤ | 日间情绪趋势 | Layer 1 (数学) | 线性回归斜率 → 改善中/稳定/下降中 |
| ⑥ | 唤醒度异常指数 | Layer 2 (LLM) | 基于arousal时间序列波动推理 |
| ⑦ | 情绪恢复速度 | Layer 1 (数学) | 负面→正面转换频次/总间隔 |
| ⑧ | 睡眠质量预测 | Layer 2 (LLM) | 基于晚间情绪 + arousal模式推理 |
| ⑨ | 压力累积指数 | Layer 2 (LLM) | 综合负面累积 + 恢复速度 + 趋势推理 |
| ⑩ | 积极情绪占比 | Layer 1 (数学) | 正面情绪记录数 / 总记录数 |
| ⑪ | 情绪突变检测 | Layer 1 (数学) | 相邻分数变化 > 30% 计数 |
| ⑫ | 综合心理健康评分 | Layer 2 (LLM) | 加权综合所有指标 + 专业判断 |

---

## 降级与容错

```
├── LLM 调用成功
│   └── 以 LLM 推理值覆盖 7 项指标
│   └── model_version = "ZhipuGLM-4-Flash-v1.0"
│
├── LLM 调用失败（网络/API异常/JSON解析失败）
│   └── 自动回退 rule-engine
│   └── _lstm_transformer_analysis()  模拟 LSTM-Transformer 预测
│   └── _determine_risk_level()       规则引擎判定风险等级
│   └── _generate_suggestions()       模板生成建议
│   └── model_version = "RuleEngine-Fallback-v2.1"
│
└── 数据不足（无情绪记录）
    └── _default_indicators()         返回基线默认值
```

**关键保证**：无论 LLM 是否可用，12项指标始终完整输出，不会阻断下游 alert → feedback 流程。

---

## 测试验证

### 测试 AI 连通性（心理健康分析）

```bash
cd "d:/Personal/projects/DaChuang-project"
python -c "
from backend.llm.zhipu import generate_mental_health_analysis
result = generate_mental_health_analysis(
    student_id=9999,
    baseline=0.7, avg_emotion=0.65, variance=0.02, entropy=0.3,
    trend_slope=0.01, trend='稳定', abrupt_count=1,
    recovery_speed=0.5, recovery_count=2, avg_recovery_interval='6.0h',
    positive_ratio=0.6, negative_ratio=0.15,
    emotion_distribution={'开心': 5, '平静': 3, '焦虑': 1},
    valence_series=[0.6, 0.7, 0.65, 0.7, 0.8],
    arousal_series=[0.3, 0.2, 0.3, 0.25, 0.2],
    record_count=10, analysis_window_days=7
)
print('AI心理健康分析调用成功！')
print('风险等级:', result['risk_assessment']['level'])
print('综合评分:', result['indicators']['overall_mental_health_score'])
"
```

### 测试容错降级（模拟 LLM 不可用）

```bash
cd "d:/Personal/projects/DaChuang-project"
python -c "
from backend.tools.mental_health import MentalHealthAnalysisTool

tool = MentalHealthAnalysisTool()
result = tool.execute(
    student_id=1,
    records=[{'fused_score': 0.7, 'fused_emotion': '开心', 'fused_valence': 0.6, 'fused_arousal': 0.2}],
    baseline=0.65,
)
print('降级测试通过！' if result.success else '失败: ' + result.error)
print('模型版本:', result.data.get('model_version'))
print('风险等级:', result.data.get('risk_level'))
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

**Q: LLM 推理的7项指标和原规则引擎的值差别大吗？**

A: 差别较大是正常的。LLM 会综合所有数学事实进行多维推理，而非简单的阈值判断。例如，同样的负面占比，如果恢复速度快、趋势改善，LLM 可能判定风险较低。如需对比，可在返回结果中查看 `model_version`（"ZhipuGLM-4-Flash-v1.0" vs "RuleEngine-Fallback-v2.1"）。

**Q: 如何调整 LLM 推理风格？**

A: 编辑 `backend/llm/zhipu.py` 中 `generate_mental_health_analysis()` 的 `system_prompt`，调整评估标准、保守/激进程度等。

**Q: 如何控制 LLM 输出质量？**

A: 当前参数 `temperature=0.3` 保证稳定性，`response_format={"type": "json_object"}` 保证结构化输出。如需更严格的指标范围约束，可在 `user_prompt` 中补充更详细的评分锚定规则。

**Q: API 调用失败会影响系统运行吗？**

A: 不会。`execute()` 内部对 LLM 调用做了 try-except 包裹，失败时自动降级为 `RuleEngine-Fallback-v2.1`，12项指标、风险等级、建议均能正常产出。

**Q: 如何更换为大模型？**

A: 与反馈模块共用同一配置，修改 `backend/config.py` 中的 `zhipu_base_url` 和 `zhipu_model` 即可。由于使用 OpenAI 兼容接口，可无缝切换其他兼容厂商。
