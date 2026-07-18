<template>
  <div>
    <div class="card">
      <h3>预警列表</h3>
      <div class="filter-bar">
        <select v-model="filterSeverity" @change="load">
          <option value="">全部等级</option>
          <option value="green">🟢 绿色</option>
          <option value="yellow">🟡 黄色</option>
          <option value="red">🔴 红色</option>
        </select>
        <select v-model="filterAck" @change="load">
          <option :value="null">全部状态</option>
          <option :value="0">未确认</option>
          <option :value="1">已确认</option>
        </select>
      </div>

      <!-- 预警表格 -->
      <table>
        <thead>
          <tr>
            <th>等级</th><th>学生</th><th>原因</th><th>反馈渠道</th><th>时间</th><th>操作</th>
          </tr>
        </thead>
        <tbody>
          <template v-for="a in alerts" :key="a.id">
            <!-- 预警主行 -->
            <tr
              :class="['alert-row', { expanded: expandedId === a.id }]"
              @click="toggleExpand(a)"
            >
              <td><span class="badge" :class="a.severity">{{ a.severity.toUpperCase() }}</span></td>
              <td>{{ getStudentName(a.student_id) }}</td>
              <td>{{ a.alert_reason }}</td>
              <td>{{ a.feedback_channel }}</td>
              <td>{{ a.triggered_at }}</td>
              <td @click.stop>
                <button v-if="!a.is_acknowledged" class="btn btn-success" @click="ack(a.id)">确认</button>
                <span v-else style="color:#999">已确认</span>
              </td>
            </tr>

            <!-- 展开的AI反馈详情 -->
            <tr v-if="expandedId === a.id" class="feedback-detail-row">
              <td colspan="6">
                <div class="feedback-detail">
                  <div class="feedback-header">
                    <span class="feedback-tag" :class="feedbackParsed(a).aiTag">
                      {{ feedbackParsed(a).aiTag === 'ai' ? '🤖 AI 生成' : '📋 模板' }}
                    </span>
                    <span class="feedback-title">{{ feedbackParsed(a).title || '通知详情' }}</span>
                  </div>

                  <!-- 各渠道通知内容 -->
                  <div class="channel-list">
                    <div
                      v-for="ch in feedbackParsed(a).channels"
                      :key="ch.key"
                      class="channel-card"
                      :class="{ highlight: ch.key === 'dashboard' }"
                    >
                      <div class="channel-label">{{ ch.icon }} {{ ch.label }}</div>
                      <div class="channel-content">{{ ch.text }}</div>
                    </div>
                  </div>

                  <!-- 无数据时的提示 -->
                  <div v-if="feedbackParsed(a).channels.length === 0" class="no-feedback">
                    暂无详细反馈内容。请触发外环分析后查看。
                  </div>
                </div>
              </td>
            </tr>
          </template>
        </tbody>
      </table>

      <div v-if="alerts.length === 0" style="text-align:center;padding:30px;color:#999">
        暂无预警记录
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { fetchAlerts, fetchStudents, acknowledgeAlert } from "../api";

const alerts = ref([]);
const students = ref([]);
const filterSeverity = ref("");
const filterAck = ref(null);
const expandedId = ref(null);

onMounted(async () => {
  students.value = await fetchStudents();
  await load();
});

async function load() {
  alerts.value = await fetchAlerts(filterSeverity.value || undefined, filterAck.value);
}

function getStudentName(id) {
  const s = students.value.find(st => st.id === id);
  return s ? s.name : `学生#${id}`;
}

function toggleExpand(alert) {
  expandedId.value = expandedId.value === alert.id ? null : alert.id;
}

function feedbackParsed(alert) {
  try {
    const data = JSON.parse(alert.feedback_content);
    const channels = [];
    const labelMap = {
      dashboard:        { label: "看板",          icon: "📊" },
      app_push:         { label: "APP 推送",      icon: "📱" },
      wechat_teacher:   { label: "微信（班主任）",  icon: "💬" },
      wechat_parent:    { label: "微信（家长）",    icon: "👨‍👩‍👧" },
      sms_parent:       { label: "短信（家长）",    icon: "✉️" },
      email_psychologist: { label: "邮件（心理教师）", icon: "📧" },
    };
    for (const [key, info] of Object.entries(labelMap)) {
      if (data[key]) {
        channels.push({ key, label: info.label, icon: info.icon, text: data[key] });
      }
    }
    return {
      title: data.title || "",
      channels,
      aiTag: data.ai_generated ? "ai" : "template",
    };
  } catch {
    // 旧格式纯文本
    return {
      title: "",
      channels: alert.feedback_content
        ? [{ key: "dashboard", label: "通知内容", icon: "📋", text: alert.feedback_content }]
        : [],
      aiTag: "template",
    };
  }
}

async function ack(id) {
  await acknowledgeAlert(id);
  await load();
}
</script>

<style scoped>
/* 预览行可点击 */
.alert-row { cursor: pointer; }
.alert-row:hover { background: #f5f7fa; }
.alert-row.expanded { background: #eef1f6; }

/* 反馈详情行 */
.feedback-detail-row { background: #fafbfc; }
.feedback-detail-row td { padding: 0; }
.feedback-detail {
  padding: 16px 20px;
  border-top: 1px dashed #ddd;
}

.feedback-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}
.feedback-tag {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 4px;
}
.feedback-tag.ai    { background: #e8f5e9; color: #2e7d32; }
.feedback-tag.template { background: #fff3e0; color: #e65100; }
.feedback-title { font-weight: 600; font-size: 15px; }

.channel-list {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.channel-card {
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 10px 14px;
}
.channel-card.highlight { border-color: #1890ff; background: #f0f7ff; }
.channel-label { font-size: 12px; color: #666; margin-bottom: 4px; font-weight: 600; }
.channel-content { font-size: 13px; color: #333; line-height: 1.5; white-space: pre-wrap; }
.no-feedback { text-align: center; color: #999; padding: 16px; font-size: 13px; }

@media (max-width: 800px) { .channel-list { grid-template-columns: 1fr; } }
</style>
