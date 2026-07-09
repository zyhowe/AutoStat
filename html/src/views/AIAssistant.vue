<template>
  <div class="ai-assistant">
    <div class="workbench-body">
      <!-- ===== 左侧：工具面板 ===== -->
      <div class="tool-panel-wrapper">
        <ToolPanel
          ref="toolPanelRef"
          :scene="currentScene"
          :session-id="sessionStore.currentSessionId"
          :personalized-questions="personalizedQuestions"
          @question-click="handleQuestionClick"
          @refresh-questions="loadRecommendedQuestions"
        />
      </div>

      <!-- ===== 中间：对话区域 ===== -->
      <div class="chat-wrapper">
        <ChatArea
          ref="chatAreaRef"
          :messages="chatStore.messages"
          :is-streaming="chatStore.isStreaming"
          :streaming-content="chatStore.streamingContent"
          :pending-tool="pendingTool"
          :context-value="currentContexts"
          @send="handleSend"
          @clear="handleClear"
          @tool-clear="pendingTool = null"
          @context-change="handleContextChange"
        />
      </div>
    </div>

    <!-- ===== 底部工具栏 ===== -->
    <div class="footer-toolbar">
      <el-button size="small" text @click="handleClear">🗑️ 清空对话</el-button>
      <el-button size="small" text @click="handleExport">📥 导出对话</el-button>
      <el-tag v-if="chatStore.messages.length > 0" size="small" type="info">
        {{ chatStore.messages.length }} 条消息
      </el-tag>
      <el-tag v-if="currentContexts.length > 0 && currentContexts.length < 3" size="small" type="warning">
        上下文: {{ getContextLabels().join('、') }}
      </el-tag>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { useChatStore } from '../stores/chat'
import { reportApi } from '../api/report'
import api from '../api'
import ToolPanel from '../components/ai/ToolPanel.vue'
import ChatArea from '../components/ai/ChatArea.vue'

const router = useRouter()
const route = useRoute()
const sessionStore = useSessionStore()
const chatStore = useChatStore()

// ===== Refs =====
const toolPanelRef = ref(null)
const chatAreaRef = ref(null)
const pendingTool = ref(null)
const loadingContext = ref(false)
const reportData = ref(null)
const personalizedQuestions = ref({})
const activeToolId = ref('')

// ===== 上下文状态 =====
const currentContexts = ref(['json'])
const isManualOverride = ref(false)

// ===== 场景映射 =====
const sceneMap = {
  '/report-summary': 'report_summary',
  '/data-overview': 'data_overview',
  '/quality': 'quality',
  '/data-validation': 'data_validation',
  '/pattern-discovery': 'pattern_discovery',
  '/models': 'smart_prediction'
}

// ===== 计算属性 =====
const currentScene = computed(() => {
  const path = route.path
  return sceneMap[path] || 'general'
})

const dataShape = computed(() => {
  const session = sessionStore.currentSession
  return session?.data_shape || { rows: 0, columns: 0 }
})

const variableTypes = computed(() => {
  const session = sessionStore.currentSession
  return session?.variable_types || {}
})

const qualityScore = computed(() => {
  return reportData.value?.overall_score ?? null
})

const dimensionScores = computed(() => {
  return reportData.value?.dimensions || {}
})

const fieldList = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  return Object.entries(summaries).map(([name, info]) => ({
    name,
    type: info.type_desc || info.type,
    count: info.count || 0,
    missing_pct: info.missing_pct || 0
  }))
})

const correlationPairs = computed(() => {
  return reportData.value?.correlations?.high_correlations || []
})

const tsFields = computed(() => {
  const diag = reportData.value?.time_series_diagnostics || {}
  return Object.entries(diag).map(([name, info]) => ({
    name,
    has_autocorrelation: info.has_autocorrelation || false,
    is_stationary: info.is_stationary || false
  }))
})

const categoricalFields = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  const result = []
  for (const [name, info] of Object.entries(summaries)) {
    if (info.type === 'categorical' || info.type === 'categorical_numeric' || info.type === 'ordinal') {
      result.push({
        name,
        n_unique: info.n_unique || 0,
        top_pct: info.mode_pct || 0
      })
    }
  }
  return result
})

const outlierFields = computed(() => {
  const outliers = reportData.value?.outliers || {}
  return Object.entries(outliers).map(([name, info]) => ({
    name,
    count: info.count || 0,
    percent: info.percent || 0
  }))
})

const missingFields = computed(() => {
  return reportData.value?.missing || []
})

const auditRules = computed(() => {
  const rules = reportData.value?.audit_rules || {}
  return [
    ...(rules.arithmetic_rules || []),
    ...(rules.functional_dependencies || []),
    ...(rules.temporal_rules || [])
  ]
})

const auditRulesCount = computed(() => auditRules.value.length)
const auditViolationsCount = computed(() => {
  return auditRules.value.reduce((sum, r) => sum + (r.violation_count || 0), 0)
})
const auditSatisfyRate = computed(() => {
  if (auditRulesCount.value === 0) return 100
  const total = auditRules.value.reduce((sum, r) => sum + (r.valid_rows || 0), 0)
  const satisfied = auditRules.value.reduce((sum, r) => sum + (r.satisfied_rows || 0), 0)
  return total > 0 ? Math.round((satisfied / total) * 100) : 0
})

const duplicateCount = computed(() => {
  return reportData.value?.duplicates?.count || 0
})

const duplicateRate = computed(() => {
  return reportData.value?.duplicates?.percent || 0
})

const previewData = computed(() => {
  return sessionStore.previewData || []
})

const previewColumns = computed(() => {
  return Object.keys(variableTypes.value)
})

const modelRecommendations = computed(() => {
  return reportData.value?.model_recommendations || []
})

const trainedModels = computed(() => {
  return reportData.value?.trained_models || []
})

const conclusions = computed(() => {
  return reportData.value?.summary || []
})

const insightsList = computed(() => {
  return reportData.value?.insights?.findings || []
})

// ===== 上下文辅助方法 =====
function getContextLabels() {
  return currentContexts.value.map(c => {
    const labels = { json: 'JSON 结果', upload: '上传数据', source: '源数据' }
    return labels[c] || c
  })
}

// ===== 处理点击推荐问题 =====
function handleQuestionClick(payload) {
  // 直接使用 ToolPanel 传过来的 text（已经是 prompt）和 dataKey
  const { text, dataKey } = payload
  const contextData = {
    dataKey: dataKey,
    scene: currentScene.value
  }
  handleSend(text, contextData)
}

function handleSend(text, contextData = null) {
  const message = text || pendingTool.value?.prompt
  if (!message) return

  const payload = {
    question: message,
    context_data: contextData || {}
  }

  if (message.includes('预测') || message.includes('forecast') || message.includes('未来趋势')) {
    chatStore.sendPrediction(
      message,
      null,
      () => { pendingTool.value = null },
      (err) => ElMessage.error(err)
    )
  } else {
    chatStore.sendMessageWithContext(
      payload,
      null,
      () => { pendingTool.value = null },
      (err) => ElMessage.error(err)
    )
  }
}

function handleClear() {
  chatStore.clearHistory()
  ElMessage.success('对话已清空')
}

function handleExport() {
  chatStore.downloadMarkdown()
  ElMessage.success('导出成功')
}

function handleContextChange(contexts) {
  currentContexts.value = contexts
  isManualOverride.value = true
}

async function loadRecommendedQuestions() {
  try {
    const sessionId = sessionStore.currentSessionId
    if (!sessionId) {
      personalizedQuestions.value = {}
      return
    }

    const response = await api.get(`/session/${sessionId}/recommended_questions`)
    const data = response.data || response

    if (data && data.questions) {
      personalizedQuestions.value = data.questions
    } else if (data && typeof data === 'object') {
      personalizedQuestions.value = data
    } else {
      personalizedQuestions.value = {}
    }
  } catch (err) {
    console.warn('加载个性化推荐失败:', err)
    personalizedQuestions.value = {}
  }
}

async function refreshContext() {
  loadingContext.value = true
  try {
    const sessionId = sessionStore.currentSessionId
    if (sessionId) {
      const result = await reportApi.get(sessionId)
      reportData.value = result
    }
  } catch (err) {
    console.warn('刷新上下文失败:', err)
  } finally {
    loadingContext.value = false
  }
}

async function init() {
  const sessionId = sessionStore.currentSessionId
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  chatStore.init(sessionId)
  await refreshContext()
  await loadRecommendedQuestions()
}

watch(() => sessionStore.currentSessionId, (newId) => {
  if (newId) {
    chatStore.init(newId)
    refreshContext()
    loadRecommendedQuestions()
  }
})

watch(() => route.path, () => {
  loadRecommendedQuestions()
})

onMounted(() => {
  init()
})
</script>

<style scoped>
.ai-assistant {
  display: flex;
  flex-direction: column;
  height: 100%;
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 16px;
}

.workbench-body {
  display: flex;
  flex: 1;
  min-height: 0;
  gap: 0;
  margin-top: 0;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}

.tool-panel-wrapper {
  width: 260px;
  flex-shrink: 0;
  height: 100%;
  overflow: hidden;
}

.chat-wrapper {
  flex: 1;
  height: 100%;
  overflow: hidden;
}

.footer-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-top: 1px solid #e4e7ed;
  flex-shrink: 0;
  margin-top: 8px;
}

@media (max-width: 992px) {
  .tool-panel-wrapper {
    width: 200px;
  }
}

@media (max-width: 768px) {
  .workbench-body {
    flex-direction: column;
  }
  .tool-panel-wrapper {
    width: 100%;
    height: auto;
    max-height: 200px;
    overflow-y: auto;
  }
  .chat-wrapper {
    height: 400px;
    min-height: 300px;
  }
  .ai-assistant {
    padding: 0 8px;
  }
}
</style>