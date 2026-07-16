<template>
  <div class="ai-assistant">
    <div class="workbench-body">
      <div class="chat-wrapper">
        <ChatArea
          ref="chatAreaRef"
          :messages="chatStore.messages"
          :is-streaming="chatStore.isStreaming"
          :streaming-content="chatStore.streamingContent"
          :pending-tool="pendingTool"
          :context-value="currentContext"
          @send="handleSend"
          @clear="handleClear"
          @tool-clear="pendingTool = null"
          @context-change="handleContextChange"
        />
      </div>

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
    </div>

    <div class="footer-toolbar">
      <el-button size="small" text @click="handleClear">🗑️ 清空对话</el-button>
      <el-button size="small" text @click="handleExport">📥 导出对话</el-button>
      <el-tag v-if="chatStore.messages.length > 0" size="small" type="info">
        {{ chatStore.messages.length }} 条消息
      </el-tag>
      <el-tag v-if="currentContext === 'upload'" size="small" type="warning">
        📁 上传数据
      </el-tag>
      <el-tag v-else-if="currentContext === 'source'" size="small" type="warning">
        🗄️ 源数据
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

const toolPanelRef = ref(null)
const chatAreaRef = ref(null)
const pendingTool = ref(null)
const loadingContext = ref(false)
const reportData = ref(null)
const personalizedQuestions = ref({})
const activeToolId = ref('')

const currentContext = ref('upload')

const sceneMap = {
  '/report-summary': 'report_summary',
  '/data-overview': 'data_overview',
  '/quality': 'quality',
  '/data-validation': 'data_validation',
  '/pattern-discovery': 'pattern_discovery',
  '/models': 'smart_prediction'
}

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

// ===== 推荐问题去重合并（从 all_tables 读取，作为备用） =====
const allQuestionsMerged = computed(() => {
  if (!reportData.value?.all_tables) return {}

  const allQuestions = {}
  const seenTexts = new Set()

  for (const [tableName, tableData] of Object.entries(reportData.value.all_tables)) {
    const questions = tableData.recommended_questions || {}
    for (const [scene, qList] of Object.entries(questions)) {
      if (!allQuestions[scene]) {
        allQuestions[scene] = []
      }
      for (const q of qList) {
        const text = q.text || ''
        if (text && !seenTexts.has(text)) {
          seenTexts.add(text)
          allQuestions[scene].push({
            ...q,
            tables: [tableName]
          })
        } else {
          // 如果已存在，追加来源表
          const existing = allQuestions[scene].find(item => item.text === text)
          if (existing && !existing.tables.includes(tableName)) {
            existing.tables.push(tableName)
          }
        }
      }
    }
  }

  return allQuestions
})

// ===== 处理点击推荐问题 =====
function handleQuestionClick(payload) {
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

  chatStore.sendMessageWithContext(
    payload,
    null,
    () => { pendingTool.value = null },
    (err) => ElMessage.error(err)
  )
}

function handleClear() {
  chatStore.clearHistory()
  ElMessage.success('对话已清空')
}

function handleExport() {
  chatStore.downloadMarkdown()
  ElMessage.success('导出成功')
}

function handleContextChange(value) {
  currentContext.value = value
}

async function loadRecommendedQuestions() {
  try {
    const sessionId = sessionStore.currentSessionId
    console.log('🔍 [调试] sessionId:', sessionId)

    if (!sessionId) {
      personalizedQuestions.value = {}
      return
    }

    const response = await api.get(`/session/${sessionId}/recommended_questions`)
    const data = response.data || response

    console.log('🔍 [调试] 原始响应 data:', JSON.stringify(data, null, 2))
    console.log('🔍 [调试] data 的 keys:', Object.keys(data))

    // ✅ 修复：后端直接返回 { merged: {}, all_tables: {} }，没有 questions 包装
    if (data && typeof data === 'object' && (data.merged || data.all_tables)) {
      const allQuestions = data  // data 本身就是 { merged: {}, all_tables: {} }
      const merged = {}
      const seenTexts = new Set()
      let totalCount = 0

      function mergeQuestion(scene, subScene, q) {
        const text = q.text || ''
        if (text && !seenTexts.has(text)) {
          seenTexts.add(text)
          if (!merged[scene]) merged[scene] = {}
          if (!merged[scene][subScene]) merged[scene][subScene] = []
          merged[scene][subScene].push(q)
          totalCount++
        }
      }

      // 处理 merged
      if (allQuestions.merged) {
        for (const [scene, sceneData] of Object.entries(allQuestions.merged)) {
          for (const [subScene, qList] of Object.entries(sceneData)) {
            for (const q of qList) {
              mergeQuestion(scene, subScene, q)
            }
          }
        }
      }

      // 处理 all_tables
      if (allQuestions.all_tables) {
        for (const [tableName, tableQuestions] of Object.entries(allQuestions.all_tables)) {
          for (const [scene, sceneData] of Object.entries(tableQuestions)) {
            for (const [subScene, qList] of Object.entries(sceneData)) {
              for (const q of qList) {
                mergeQuestion(scene, subScene, q)
              }
            }
          }
        }
      }

      console.log(`🔍 [调试] 合并完成: ${Object.keys(merged).length} 个场景，${totalCount} 条问题`)
      personalizedQuestions.value = merged
    } else {
      console.log('🔍 [调试] data 结构不符，降级到 allQuestionsMerged')
      personalizedQuestions.value = allQuestionsMerged.value
    }
  } catch (err) {
    console.error('🔍 [调试] 加载失败:', err)
    personalizedQuestions.value = allQuestionsMerged.value
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
  max-width: 100%;
  margin: 0;
  padding: 0 12px;
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

.chat-wrapper {
  flex: 1;
  height: 100%;
  overflow: hidden;
  min-width: 0;
}

.tool-panel-wrapper {
  width: 280px;
  flex-shrink: 0;
  height: 100%;
  overflow: hidden;
  border-left: 1px solid #e4e7ed;
  background: #fafafa;
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
    border-left: none;
    border-top: 1px solid #e4e7ed;
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