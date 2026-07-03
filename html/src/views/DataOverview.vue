<template>
  <div class="data-overview">
    <h2>📊 数据总览</h2>
    <p class="subtitle">查看数据的基本信息、字段详情，并支持数据查询</p>

    <el-tabs v-model="activeTab" class="overview-tabs">
      <!-- Tab1: 数据概览 -->
      <el-tab-pane label="数据概览" name="overview">
        <div v-if="loading" class="loading-container">
          <el-skeleton :rows="10" animated />
        </div>
        <div v-else-if="sessionStore.hasSession && sessionStore.currentSession">
          <div class="stats-row">
            <div class="stat-card">
              <div class="stat-value">{{ sessionStore.currentSession.data_shape?.rows || 0 }}</div>
              <div class="stat-label">总行数</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">{{ sessionStore.currentSession.data_shape?.columns || 0 }}</div>
              <div class="stat-label">总列数</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">{{ variableTypesCount }}</div>
              <div class="stat-label">变量类型</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">{{ missingFieldsCount }}</div>
              <div class="stat-label">缺失字段</div>
            </div>
          </div>

          <!-- 变量类型分布 -->
          <div class="section">
            <h4>📋 变量类型分布</h4>
            <div class="type-tags">
              <el-tag
                v-for="(count, type) in typeCounts"
                :key="type"
                size="large"
                class="type-tag"
              >
                {{ typeDisplay[type] || type }}：{{ count }}
              </el-tag>
            </div>
          </div>

          <!-- 变量详情表格 -->
          <div class="section">
            <h4>📋 字段详情</h4>
            <el-table :data="variableList" border size="small" max-height="400">
              <el-table-column prop="name" label="字段名" width="150" />
              <el-table-column prop="type_desc" label="类型" width="120" />
              <el-table-column prop="count" label="样本量" width="100" />
              <el-table-column prop="missing" label="缺失数" width="100" />
              <el-table-column prop="missing_pct" label="缺失率" width="100">
                <template #default="{ row }">
                  {{ row.missing_pct.toFixed(1) }}%
                </template>
              </el-table-column>
              <el-table-column prop="center" label="中心趋势" width="120" />
              <el-table-column prop="spread" label="分布" />
            </el-table>
          </div>
        </div>
        <div v-else class="empty-state">
          <el-empty description="请先上传数据并完成分析">
            <el-button type="primary" @click="goToUpload">去上传数据</el-button>
          </el-empty>
        </div>
      </el-tab-pane>

      <!-- Tab2: 数据查询 -->
      <el-tab-pane label="数据查询" name="query">
        <div class="query-container">
          <div class="query-modes">
            <el-radio-group v-model="queryMode" size="large">
              <el-radio-button value="natural">自然语言查询</el-radio-button>
              <el-radio-button value="sql">SQL 生成</el-radio-button>
            </el-radio-group>
          </div>

          <!-- 自然语言查询 -->
          <div v-if="queryMode === 'natural'" class="query-panel">
            <div class="chat-box">
              <div class="chat-messages" ref="chatContainer">
                <div v-if="nlMessages.length === 0" class="chat-empty">
                  用自然语言提问，AI 帮你分析数据
                </div>
                <div
                  v-for="(msg, index) in nlMessages"
                  :key="index"
                  class="chat-msg"
                  :class="msg.role"
                >
                  <span class="msg-avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</span>
                  <span class="msg-content">{{ msg.content }}</span>
                </div>
                <div v-if="nlStreaming" class="chat-msg assistant">
                  <span class="msg-avatar">🤖</span>
                  <span class="msg-content">{{ nlStreamingContent }}<span class="cursor">▌</span></span>
                </div>
              </div>
              <div class="chat-input">
                <el-input
                  v-model="nlQuestion"
                  placeholder="输入查询，如：查询销售额最高的前10条"
                  @keyup.enter="sendNaturalQuery"
                >
                  <template #append>
                    <el-button type="primary" :loading="nlStreaming" @click="sendNaturalQuery">
                      查询
                    </el-button>
                  </template>
                </el-input>
              </div>
              <div class="quick-questions">
                <span class="label">快速提问：</span>
                <el-button
                  v-for="q in naturalExamples"
                  :key="q"
                  size="small"
                  @click="nlQuestion = q; sendNaturalQuery()"
                >
                  {{ q }}
                </el-button>
              </div>
            </div>
          </div>

          <!-- SQL生成 -->
          <div v-if="queryMode === 'sql'" class="query-panel">
            <div class="chat-box">
              <div class="chat-messages" ref="sqlChatContainer">
                <div v-if="sqlMessages.length === 0" class="chat-empty">
                  描述你的 SQL 需求，AI 帮你生成 SQL
                </div>
                <div
                  v-for="(msg, index) in sqlMessages"
                  :key="index"
                  class="chat-msg"
                  :class="msg.role"
                >
                  <span class="msg-avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</span>
                  <span class="msg-content">{{ msg.content }}</span>
                </div>
                <div v-if="sqlStreaming" class="chat-msg assistant">
                  <span class="msg-avatar">🤖</span>
                  <span class="msg-content">{{ sqlStreamingContent }}<span class="cursor">▌</span></span>
                </div>
              </div>
              <div class="chat-input">
                <el-input
                  v-model="sqlQuestion"
                  placeholder="描述SQL需求，如：查询最近7天的数据"
                  @keyup.enter="sendSqlQuery"
                >
                  <template #append>
                    <el-button type="primary" :loading="sqlStreaming" @click="sendSqlQuery">
                      生成
                    </el-button>
                  </template>
                </el-input>
              </div>
              <div class="quick-questions">
                <span class="label">快速示例：</span>
                <el-button
                  v-for="q in sqlExamples"
                  :key="q"
                  size="small"
                  @click="sqlQuestion = q; sendSqlQuery()"
                >
                  {{ q }}
                </el-button>
              </div>
            </div>
          </div>
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { chatApi } from '../api/chat'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(false)
const activeTab = ref('overview')
const queryMode = ref('natural')

// 数据概览
const typeDisplay = {
  continuous: '连续变量',
  categorical: '分类变量',
  categorical_numeric: '数值型分类',
  ordinal: '有序分类',
  datetime: '日期时间',
  identifier: '标识符',
  text: '文本'
}

const typeCounts = computed(() => {
  const session = sessionStore.currentSession
  if (!session?.variable_types) return {}
  const counts = {}
  for (const info of Object.values(session.variable_types)) {
    const typ = info.type || 'unknown'
    counts[typ] = (counts[typ] || 0) + 1
  }
  return counts
})

const variableTypesCount = computed(() => Object.keys(typeCounts.value).length)
const missingFieldsCount = computed(() => {
  const session = sessionStore.currentSession
  return session?.analysis_result?.quality_report?.missing?.length || 0
})

const variableList = computed(() => {
  const session = sessionStore.currentSession
  if (!session?.analysis_result?.variable_summaries) return []
  const summaries = session.analysis_result.variable_summaries
  return Object.entries(summaries).map(([name, info]) => ({
    name,
    type_desc: info.type_desc || info.type,
    count: info.count || 0,
    missing: info.missing || 0,
    missing_pct: info.missing_pct || 0,
    center: info.mean !== undefined ? info.mean.toFixed(2) : (info.mode || '-'),
    spread: info.std !== undefined ? `±${info.std.toFixed(2)}` : (info.n_unique ? `${info.n_unique}个类别` : '-')
  })).slice(0, 30)
})

function goToUpload() {
  router.push('/upload')
}

// 自然语言查询
const nlMessages = ref([])
const nlQuestion = ref('')
const nlStreaming = ref(false)
const nlStreamingContent = ref('')
const chatContainer = ref(null)

const naturalExamples = [
  '查询销售额最高的前10条',
  '统计各分类的分布',
  '查询最近7天的数据',
  '找出异常值'
]

async function sendNaturalQuery() {
  const q = nlQuestion.value.trim()
  if (!q) return
  if (!sessionStore.hasSession) {
    ElMessage.warning('请先完成数据分析')
    return
  }

  nlMessages.value.push({ role: 'user', content: q })
  nlQuestion.value = ''
  nlStreaming.value = true
  nlStreamingContent.value = ''

  try {
    await chatApi.chatStream(
      sessionStore.currentSessionId,
      `请查询数据: ${q}`,
      ['json_result', 'raw_data'],
      (chunk) => {
        nlStreamingContent.value += chunk
        scrollToBottom(chatContainer)
      },
      () => {
        nlMessages.value.push({ role: 'assistant', content: nlStreamingContent.value })
        nlStreamingContent.value = ''
        nlStreaming.value = false
        scrollToBottom(chatContainer)
      },
      (error) => {
        ElMessage.error('查询失败: ' + error)
        nlStreaming.value = false
      }
    )
  } catch (err) {
    ElMessage.error('查询失败: ' + err.message)
    nlStreaming.value = false
  }
}

// SQL生成
const sqlMessages = ref([])
const sqlQuestion = ref('')
const sqlStreaming = ref(false)
const sqlStreamingContent = ref('')
const sqlChatContainer = ref(null)

const sqlExamples = [
  '查询最近7天的数据',
  '按分类统计数量',
  '关联查询两张表',
  '查询最大值'
]

async function sendSqlQuery() {
  const q = sqlQuestion.value.trim()
  if (!q) return
  if (!sessionStore.hasSession) {
    ElMessage.warning('请先完成数据分析')
    return
  }

  sqlMessages.value.push({ role: 'user', content: `请生成SQL: ${q}` })
  sqlQuestion.value = ''
  sqlStreaming.value = true
  sqlStreamingContent.value = ''

  try {
    await chatApi.chatStream(
      sessionStore.currentSessionId,
      `请生成SQL: ${q}`,
      ['json_result'],
      (chunk) => {
        sqlStreamingContent.value += chunk
        scrollToBottom(sqlChatContainer)
      },
      () => {
        sqlMessages.value.push({ role: 'assistant', content: sqlStreamingContent.value })
        sqlStreamingContent.value = ''
        sqlStreaming.value = false
        scrollToBottom(sqlChatContainer)
      },
      (error) => {
        ElMessage.error('生成失败: ' + error)
        sqlStreaming.value = false
      }
    )
  } catch (err) {
    ElMessage.error('生成失败: ' + err.message)
    sqlStreaming.value = false
  }
}

function scrollToBottom(ref) {
  nextTick(() => {
    if (ref?.value) {
      ref.value.scrollTop = ref.value.scrollHeight
    }
  })
}
</script>

<style scoped>
.data-overview {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}
.subtitle {
  color: #909399;
  margin-bottom: 24px;
}
.loading-container {
  padding: 40px 0;
}
.empty-state {
  padding: 60px 0;
}

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 30px;
}
.stat-card {
  background: #f5f7fa;
  border-radius: 12px;
  padding: 20px;
  text-align: center;
}
.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: #2c3e50;
}
.stat-label {
  font-size: 13px;
  color: #909399;
  margin-top: 4px;
}

.section {
  margin-bottom: 30px;
}
.section h4 {
  margin-bottom: 12px;
  color: #2c3e50;
  font-size: 16px;
}
.type-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.type-tag {
  font-size: 14px;
  padding: 8px 16px;
}

.overview-tabs {
  margin-top: 10px;
}

/* 查询样式 */
.query-container {
  padding: 10px 0;
}
.query-modes {
  margin-bottom: 20px;
}
.query-panel {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  overflow: hidden;
}
.chat-box {
  display: flex;
  flex-direction: column;
  height: 450px;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background: #fafafa;
}
.chat-empty {
  text-align: center;
  color: #bbb;
  padding: 60px 0;
  font-size: 14px;
}
.chat-msg {
  display: flex;
  gap: 10px;
  margin-bottom: 12px;
  align-items: flex-start;
}
.chat-msg.user {
  flex-direction: row-reverse;
}
.chat-msg.user .msg-content {
  background: #409eff;
  color: white;
  padding: 8px 14px;
  border-radius: 12px 12px 4px 12px;
  max-width: 75%;
}
.chat-msg.assistant .msg-content {
  background: white;
  padding: 8px 14px;
  border-radius: 12px 12px 12px 4px;
  max-width: 75%;
  border: 1px solid #e4e7ed;
  word-break: break-word;
}
.msg-avatar {
  font-size: 18px;
  flex-shrink: 0;
}
.msg-content {
  font-size: 14px;
  line-height: 1.6;
}
.cursor {
  animation: blink 0.8s infinite;
}
@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.chat-input {
  padding: 12px 16px;
  border-top: 1px solid #e4e7ed;
  background: white;
  flex-shrink: 0;
}
.chat-input .el-input {
  width: 100%;
}

.quick-questions {
  padding: 8px 16px 12px 16px;
  background: #fafafa;
  border-top: 1px solid #f0f0f0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
.quick-questions .label {
  font-size: 12px;
  color: #909399;
  margin-right: 4px;
}
</style>