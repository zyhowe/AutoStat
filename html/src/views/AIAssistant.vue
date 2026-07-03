<template>
  <div class="ai-assistant">
    <h2>🧠 AI助手</h2>
    <p class="subtitle">智能问答与场景探索</p>

    <el-row :gutter="24">
      <!-- ==================== 左侧面板 ==================== -->
      <el-col :span="8">
        <div class="left-panel">
          <!-- 数据特征 -->
          <div class="section">
            <h4>📊 当前数据特征</h4>
            <div v-if="!sessionStore.hasSession" class="empty-feature">
              <el-empty description="请先完成数据分析" :image-size="60" />
            </div>
            <div v-else-if="dataFeatures" class="feature-grid">
              <div class="feature-item">
                <span class="label">数据量</span>
                <span class="value">{{ dataFeatures.rows }} × {{ dataFeatures.cols }}</span>
              </div>
              <div class="feature-item">
                <span class="label">变量类型</span>
                <span class="value">{{ dataFeatures.typeSummary }}</span>
              </div>
              <div class="feature-item">
                <span class="label">时间序列</span>
                <span class="value">{{ dataFeatures.tsSummary }}</span>
              </div>
              <div class="feature-item">
                <span class="label">异常值</span>
                <span class="value">{{ dataFeatures.outlierSummary }}</span>
              </div>
              <div class="feature-item">
                <span class="label">缺失值</span>
                <span class="value">{{ dataFeatures.missingSummary }}</span>
              </div>
            </div>
          </div>

          <!-- 上下文选择 -->
          <div class="section">
            <h4>📚 选择分析上下文</h4>
            <el-checkbox-group v-model="selectedContexts">
              <el-checkbox value="json_result">📊 JSON 结果</el-checkbox>
              <el-checkbox value="html_report">📄 HTML 报告</el-checkbox>
              <el-checkbox value="raw_data">🗃️ 源数据</el-checkbox>
            </el-checkbox-group>
          </div>

          <!-- 推荐问题 -->
          <div class="section">
            <h4>💡 推荐问题</h4>
            <div class="button-group">
              <el-button
                v-for="(q, index) in recommendedQuestions"
                :key="index"
                class="btn-left"
                @click="sendQuestion(q)"
              >
                {{ q }}
              </el-button>
            </div>
          </div>

          <!-- 场景推荐 -->
          <div class="section">
            <h4>🎯 场景推荐</h4>
            <div class="button-group">
              <el-button
                v-for="(s, index) in scenarios"
                :key="index"
                class="btn-left"
                @click="sendQuestion(s.question)"
              >
                {{ s.label }}
              </el-button>
            </div>
          </div>

          <!-- 自然查询 -->
          <div class="section">
            <h4 @click="toggleSection('natural')" style="cursor: pointer;">
              🔍 自然查询
              <span class="toggle-icon">{{ expandedSections.natural ? '▼' : '▶' }}</span>
            </h4>
            <div v-show="expandedSections.natural" class="section-content">
              <div class="example-buttons">
                <el-button
                  v-for="ex in naturalExamples"
                  :key="ex"
                  size="small"
                  class="btn-left"
                  @click="sendQuestion(ex)"
                >
                  {{ ex }}
                </el-button>
              </div>
              <el-input
                v-model="naturalInput"
                placeholder="输入查询..."
                size="small"
                @keyup.enter="sendNaturalQuery"
              >
                <template #append>
                  <el-button size="small" @click="sendNaturalQuery">发送</el-button>
                </template>
              </el-input>
            </div>
          </div>

          <!-- SQL生成 -->
          <div class="section">
            <h4 @click="toggleSection('sql')" style="cursor: pointer;">
              📝 SQL生成
              <span class="toggle-icon">{{ expandedSections.sql ? '▼' : '▶' }}</span>
            </h4>
            <div v-show="expandedSections.sql" class="section-content">
              <div class="example-buttons">
                <el-button
                  v-for="ex in sqlExamples"
                  :key="ex"
                  size="small"
                  class="btn-left"
                  @click="sendQuestion('请生成SQL: ' + ex)"
                >
                  {{ ex }}
                </el-button>
              </div>
              <el-input
                v-model="sqlInput"
                placeholder="描述SQL需求..."
                size="small"
                @keyup.enter="sendSqlQuery"
              >
                <template #append>
                  <el-button size="small" @click="sendSqlQuery">生成</el-button>
                </template>
              </el-input>
            </div>
          </div>

          <!-- 推理预测 -->
          <div class="section">
            <h4 @click="toggleSection('predict')" style="cursor: pointer;">
              🔮 推理预测
              <span class="toggle-icon">{{ expandedSections.predict ? '▼' : '▶' }}</span>
            </h4>
            <div v-show="expandedSections.predict" class="section-content">
              <div v-if="savedModels.length === 0" class="empty-hint">
                暂无已训练模型，请先在「模型中心」训练
              </div>
              <div v-else class="button-group">
                <el-button
                  v-for="m in savedModels"
                  :key="m.model_key"
                  size="small"
                  class="btn-left"
                  @click="sendPredictRequest(m)"
                >
                  🔮 {{ m.user_model_name || m.model_key }}
                </el-button>
              </div>
            </div>
          </div>

          <!-- 勾稽校验 -->
          <div class="section">
            <h4 @click="toggleSection('audit')" style="cursor: pointer;">
              🔗 勾稽校验
              <span class="toggle-icon">{{ expandedSections.audit ? '▼' : '▶' }}</span>
            </h4>
            <div v-show="expandedSections.audit" class="section-content">
              <div class="button-group">
                <el-button
                  size="small"
                  class="btn-left"
                  @click="sendQuestion('请解读当前数据中的勾稽规则，说明每条规则的含义和违反情况')"
                >
                  📊 解读已有规则
                </el-button>
                <el-button
                  size="small"
                  class="btn-left"
                  @click="sendQuestion('请校验当前数据的表结构，识别潜在的数据一致性问题和业务逻辑隐患')"
                >
                  📝 校验表结构
                </el-button>
                <el-button
                  size="small"
                  class="btn-left"
                  @click="sendQuestion('请分析数据中可能存在的勾稽关系，推荐可用的数据一致性规则')"
                >
                  🔍 发现潜在关系
                </el-button>
                <el-button
                  size="small"
                  class="btn-left"
                  @click="sendQuestion('请生成一份勾稽规则报告，汇总所有发现的数据一致性规则和问题')"
                >
                  📋 生成规则报告
                </el-button>
              </div>
            </div>
          </div>

        </div>
      </el-col>

      <!-- ==================== 右侧对话面板 ==================== -->
      <el-col :span="16">
        <div class="right-panel">
          <div class="chat-header">
            <h4>💬 对话</h4>
            <el-button size="small" @click="clearChat">清空</el-button>
          </div>

          <div class="chat-messages" ref="chatContainer">
            <div v-if="messages.length === 0" class="empty-chat">
              <el-empty description="从左侧选择问题开始对话" :image-size="80" />
            </div>
            <div
              v-for="(msg, index) in messages"
              :key="index"
              class="chat-message"
              :class="msg.role"
            >
              <div class="message-avatar">
                {{ msg.role === 'user' ? '👤' : '🤖' }}
              </div>
              <div class="message-content" v-html="formatMessage(msg.content)" />
            </div>
            <div v-if="isStreaming" class="chat-message assistant">
              <div class="message-avatar">🤖</div>
              <div class="message-content streaming" v-html="formatStreamingContent(streamingContent)" />
            </div>
          </div>

          <div class="chat-input">
            <el-input
              v-model="inputQuestion"
              placeholder="输入您的问题..."
              @keyup.enter="handleSend"
            >
              <template #append>
                <el-button
                  type="primary"
                  :loading="isStreaming"
                  @click="handleSend"
                >
                  {{ isStreaming ? '发送中' : '发送' }}
                </el-button>
              </template>
            </el-input>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { chatApi } from '../api/chat'
import { modelsApi } from '../api/models'

const route = useRoute()
const sessionStore = useSessionStore()

// ==================== 状态 ====================
const messages = ref([])
const inputQuestion = ref('')
const isStreaming = ref(false)
const streamingContent = ref('')
const selectedContexts = ref(['json_result'])
const recommendedQuestions = ref([])
const scenarios = ref([])
const dataFeatures = ref(null)
const chatContainer = ref(null)

// 展开状态
const expandedSections = ref({
  natural: false,
  sql: false,
  predict: false,
  audit: false
})

// 自然查询
const naturalInput = ref('')
const naturalExamples = [
  '查询最近7天的数据',
  '统计各分类的数量',
  '找出数值最大的前10条记录',
  '分析各月的变化趋势'
]

// SQL生成
const sqlInput = ref('')
const sqlExamples = [
  '查询最近7天的数据',
  '按分类统计数量',
  '关联查询两张表',
  '查询销售额最高的前10条'
]

// 推理预测
const savedModels = ref([])

// ==================== 生命周期 ====================
onMounted(async () => {
  if (sessionStore.hasSession) {
    await loadDataFeatures()
    await loadRecommendedQuestions()
    await loadScenarios()
    await loadModels()
  }

  // 监听来自气泡组件的问题
  window.addEventListener('ai-ask-question', handleAiAskQuestion)

  // 处理 URL 参数中的问题
  if (route.query.question) {
    inputQuestion.value = route.query.question
    await nextTick()
    handleSend()
  }
})

onBeforeUnmount(() => {
  window.removeEventListener('ai-ask-question', handleAiAskQuestion)
})

// ==================== 事件处理 ====================
function handleAiAskQuestion(e) {
  const question = e.detail?.question
  if (question) {
    inputQuestion.value = question
    // 清空 URL 参数，防止刷新后重复发送
    const newUrl = new URL(window.location)
    newUrl.searchParams.delete('question')
    window.history.replaceState({}, '', newUrl)
    handleSend()
  }
}

// ==================== 加载数据 ====================
async function loadDataFeatures() {
  const session = sessionStore.currentSession
  if (!session || !session.analysis_result) return

  const result = session.analysis_result
  const variableTypes = result.variable_types || {}
  const tsDiag = result.time_series_diagnostics || {}
  const quality = result.quality_report || {}

  const typeCounts = {}
  const typeDisplay = {
    continuous: '连续', categorical: '分类', datetime: '日期',
    identifier: '标识符', ordinal: '有序'
  }
  Object.values(variableTypes).forEach(info => {
    const typ = info.type || 'unknown'
    typeCounts[typ] = (typeCounts[typ] || 0) + 1
  })
  const typeSummary = Object.entries(typeCounts)
    .filter(([t]) => typeDisplay[t])
    .map(([t, c]) => `${typeDisplay[t]}${c}`).join(' / ')

  const hasAuto = Object.values(tsDiag).some(d => d.has_autocorrelation)
  const tsCount = Object.values(tsDiag).filter(d => d.has_autocorrelation).length
  const tsSummary = hasAuto ? `✅ ${tsCount}个有自相关` : '❌ 无'

  const outlierCount = Object.keys(quality.outliers || {}).length
  const missingCount = (quality.missing || []).length

  dataFeatures.value = {
    rows: result.data_shape?.rows || 0,
    cols: result.data_shape?.columns || 0,
    typeSummary: typeSummary || '无',
    tsSummary,
    outlierSummary: outlierCount > 0 ? `⚠️ ${outlierCount}个字段` : '✅ 无',
    missingSummary: missingCount > 0 ? `⚠️ ${missingCount}个字段` : '✅ 无'
  }
}

async function loadRecommendedQuestions() {
  if (!sessionStore.currentSessionId) return
  try {
    const result = await chatApi.getRecommendedQuestions(sessionStore.currentSessionId)
    recommendedQuestions.value = result || []
  } catch (err) {
    console.error('加载推荐问题失败:', err)
  }
}

async function loadScenarios() {
  if (!sessionStore.currentSessionId) return
  try {
    const result = await chatApi.getScenarios(sessionStore.currentSessionId)
    scenarios.value = result || []
  } catch (err) {
    console.error('加载场景失败:', err)
  }
}

async function loadModels() {
  if (!sessionStore.currentSessionId) return
  try {
    const result = await modelsApi.list(sessionStore.currentSessionId)
    savedModels.value = result || []
  } catch (err) {
    console.error('加载模型失败:', err)
  }
}

// ==================== 切换展开 ====================
function toggleSection(name) {
  expandedSections.value[name] = !expandedSections.value[name]
}

// ==================== 发送问题 ====================
function sendQuestion(question) {
  inputQuestion.value = question
  handleSend()
}

function sendNaturalQuery() {
  if (naturalInput.value.trim()) {
    sendQuestion(naturalInput.value.trim())
    naturalInput.value = ''
  }
}

function sendSqlQuery() {
  if (sqlInput.value.trim()) {
    sendQuestion('请生成SQL: ' + sqlInput.value.trim())
    sqlInput.value = ''
  }
}

function sendPredictRequest(model) {
  const modelName = model.user_model_name || model.model_key
  const target = model.target_column || '未知'
  const features = (model.features || []).join('、')
  const question = `请使用模型「${modelName}」进行预测，目标列是「${target}」，特征包括：${features}`
  sendQuestion(question)
}

// ==================== 对话处理 ====================
async function handleSend() {
  const question = inputQuestion.value.trim()
  if (!question) return
  if (!sessionStore.hasSession) {
    ElMessage.warning('请先完成数据分析')
    return
  }

  messages.value.push({ role: 'user', content: question })
  inputQuestion.value = ''
  scrollToBottom()

  isStreaming.value = true
  streamingContent.value = ''

  try {
    await chatApi.chatStream(
      sessionStore.currentSessionId,
      question,
      selectedContexts.value,
      (chunk) => {
        streamingContent.value += chunk
        scrollToBottom()
      },
      () => {
        messages.value.push({ role: 'assistant', content: streamingContent.value })
        streamingContent.value = ''
        isStreaming.value = false
        scrollToBottom()
      },
      (error) => {
        ElMessage.error('对话失败: ' + error)
        isStreaming.value = false
      }
    )
  } catch (err) {
    ElMessage.error('对话失败: ' + err.message)
    isStreaming.value = false
  }
}

function clearChat() {
  messages.value = []
  streamingContent.value = ''
  isStreaming.value = false
}

function scrollToBottom() {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

// ==================== 格式化消息 ====================
function formatStreamingContent(content) {
  if (!content) return ''
  return content.replace(/\n/g, '<br>') + '<span class="cursor">▌</span>'
}

function formatMessage(content) {
  if (!content) return ''
  return content
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
}
</script>

<style scoped>
.ai-assistant {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}
.subtitle {
  color: #909399;
  margin-bottom: 24px;
}

/* ===== 左侧面板 ===== */
.left-panel {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 16px;
  height: calc(100vh - 160px);
  overflow-y: auto;
}
.left-panel .section {
  margin-bottom: 16px;
}
.left-panel .section:last-child {
  margin-bottom: 0;
}
.left-panel .section h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #2c3e50;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid #e8ecf1;
  cursor: pointer;
}
.left-panel .section h4 .toggle-icon {
  font-size: 12px;
  color: #909399;
}
.section-content {
  padding: 8px 4px;
}

.feature-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 12px;
  font-size: 13px;
}
.feature-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
}
.feature-item .label {
  color: #909399;
}
.feature-item .value {
  color: #2c3e50;
  font-weight: 500;
}

/* 按钮统一容器 - 解决对齐问题 */
.button-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.button-group .btn-left,
.button-group .el-button {
  width: 100%;
  margin: 0 !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding-left: 12px;
}

.btn-left {
  width: 100%;
  margin: 0 !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding-left: 12px;
}

.question-btn {
  background: #ecf5ff;
  border-color: #d9ecff;
}
.scenario-btn {
  background: #f0f9eb;
  border-color: #e1f3d8;
}
.model-btn {
  font-size: 12px;
  background: #fff3e0;
  border-color: #ffe0b2;
}
.audit-btn {
  font-size: 12px;
  background: #f3e5f5;
  border-color: #e1bee7;
}
.example-buttons .el-button {
  justify-content: flex-start !important;
}
.empty-hint {
  font-size: 12px;
  color: #909399;
  padding: 8px 0;
}

.example-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 8px;
}
.example-buttons .el-button {
  font-size: 12px;
}

/* ===== 右侧对话面板 ===== */
.right-panel {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 160px);
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  overflow: hidden;
}
.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #e4e7ed;
  background: #fafafa;
  flex-shrink: 0;
}
.chat-header h4 {
  margin: 0;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}
.empty-chat {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}

.chat-message {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  align-items: flex-start;
}
.chat-message.user {
  flex-direction: row-reverse;
}
.chat-message.user .message-content {
  background: #409eff;
  color: white;
  border-radius: 12px 12px 4px 12px;
}
.chat-message.assistant .message-content {
  background: #f0f2f6;
  border-radius: 12px 12px 12px 4px;
}
.message-avatar {
  font-size: 20px;
  flex-shrink: 0;
  width: 32px;
  text-align: center;
}
.message-content {
  padding: 10px 14px;
  max-width: 80%;
  word-break: break-word;
  line-height: 1.8;
  font-size: 14px;
}

.message-content.streaming .cursor {
  animation: blink 0.8s infinite;
}
@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.chat-input {
  padding: 12px 16px;
  border-top: 1px solid #e4e7ed;
  background: #fafafa;
  flex-shrink: 0;
}
.chat-input .el-input {
  width: 100%;
}
</style>