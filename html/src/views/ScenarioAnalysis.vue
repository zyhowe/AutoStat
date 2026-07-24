<template>
  <div class="scenario-analysis">
    <div class="page-header">
      <h2>🔍 场景分析</h2>
      <p class="subtitle">配置场景、执行诊断、查看结论、追溯数据</p>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <!-- 错误状态 -->
    <div v-else-if="error" class="error-container">
      <el-result icon="error" :title="error" sub-title="请先完成数据分析">
        <template #extra>
          <el-button type="primary" @click="goToUpload">去上传数据</el-button>
        </template>
      </el-result>
    </div>

    <!-- 主内容 -->
    <div v-else class="analysis-container">
      <el-tabs v-model="activeTab" @tab-click="onTabChange">
        <!-- ============================================================ -->
        <!-- Tab 1: 开始分析 -->
        <!-- ============================================================ -->
        <el-tab-pane label="🚀 开始分析" name="config">
          <StartAnalysis
            ref="startAnalysisRef"
            :candidates="candidates"
            :field-mapping="fieldMapping"
            :scenario-config-saved="scenarioConfigSaved"
            :mapping-config-saved="mappingConfigSaved"
            :executing="executing"
            :exec-progress="execProgress"
            :exec-message="execMessage"
            :saving-scenarios="savingScenarios"
            :saving-mapping="savingMapping"
            :parsing="parsing"
            :enabled-count="enabledCount"
            @save-scenario-config="saveScenarioConfig"
            @reset-scenario-config="resetScenarioConfig"
            @save-mapping-config="saveMappingConfig"
            @parse-mapping="parseMapping"
            @confirm-mapping="confirmMapping"
            @remove-mapping="removeMapping"
            @clear-all-mapping="clearAllMapping"
            @load-sample-mapping="loadSampleMapping"
            @select-all="selectAll"
            @run-analysis="runScenarioAnalysis"
            @update:mapping-text="mappingText = $event"
            @update:parsed-mapping="parsedMapping = $event"
            @update:parsed-unmatched="parsedUnmatched = $event"
            @remove-parsed-mapping="removeParsedMapping"
            @clear-parsed="clearParsed"
          />
        </el-tab-pane>

        <!-- ============================================================ -->
        <!-- Tab 2: 分析摘要 -->
        <!-- ============================================================ -->
        <el-tab-pane label="📊 分析摘要" name="dashboard">
          <AnalysisSummary
            ref="analysisSummaryRef"
            :has-results="hasResults"
            :scenario-results="scenarioResults"
            :summary="summary"
            :total-records="totalRecords"
            :field-mapping="fieldMapping"
            :insights="insights"
            :loading-insights="loadingInsights"
            :expanded-card="expandedCard"
            @view-details="goToRecords"
            @toggle-expand="toggleCardDetail"
            @view-full-data="handleViewFullData"
            @go-to-config="goToConfig"
          />
        </el-tab-pane>

        <!-- ============================================================ -->
        <!-- Tab 3: 数据解码与追溯 -->
        <!-- ============================================================ -->
        <el-tab-pane label="📦 数据解码与追溯" name="records">
          <DataDecodeAndTrace
            ref="dataTraceRef"
            :scenario-results="scenarioResults"
            :all-records="allRecords"
            :field-mapping="fieldMapping"
            :session-id="sessionStore.currentSessionId || localStorage.getItem('lastSessionId')"
            :has-db-config="hasDbConfig"
            :parquet-data="parquetData"
            :trace-context="traceContext"
            @show-detail="showRecordDetail"
            @trace="handleTrace"
            @update-record-status="updateRecordStatus"
          />
        </el-tab-pane>
      </el-tabs>
    </div>

    <!-- ===== 记录详情弹窗 ===== -->
    <el-dialog v-model="detailDialogVisible" title="记录详情" width="750px" destroy-on-close>
      <div v-if="selectedRecord" class="record-detail">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="行号">{{ selectedRecord.row }}</el-descriptions-item>
          <el-descriptions-item label="所属场景">{{ selectedRecord.scenario_name }}</el-descriptions-item>
          <el-descriptions-item label="记录类型">{{ selectedRecord.record_type_display || '其他' }}</el-descriptions-item>
          <el-descriptions-item label="规则/字段">
            <span v-if="selectedRecord.record_type === 'violation'">{{ selectedRecord.rule || '—' }}</span>
            <span v-else>{{ selectedRecord.field_display || selectedRecord.field || '—' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="当前值" :span="1">
            <span v-if="selectedRecord.record_type === 'violation' && selectedRecord.values_display">
              {{ selectedRecord.values_display }}
            </span>
            <span v-else-if="selectedRecord.record_type === 'cluster'">群组 {{ selectedRecord.cluster_id }}</span>
            <span v-else>{{ selectedRecord.value_display || selectedRecord.value || '—' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="预期值/范围">
            <span v-if="selectedRecord.record_type === 'violation'">相等</span>
            <span v-else>{{ selectedRecord.expected || '—' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="偏离程度">
            <span v-if="selectedRecord.record_type === 'violation'">{{ selectedRecord.diff !== undefined ? selectedRecord.diff.toFixed(4) : '—' }}</span>
            <span v-else>{{ selectedRecord.deviation !== undefined ? selectedRecord.deviation.toFixed(2) + 'x' : '—' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="严重程度">
            <el-tag v-if="selectedRecord.severity" :type="selectedRecord.severity === 'high' ? 'danger' : selectedRecord.severity === 'medium' ? 'warning' : 'info'" size="small">{{ selectedRecord.severity }}</el-tag>
            <span v-else>—</span>
          </el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="selectedRecord.status === 'resolved' ? 'success' : selectedRecord.status === 'ignored' ? 'info' : 'warning'" size="small">
              {{ { pending: '待核查', ignored: '已忽略', resolved: '已处理' }[selectedRecord.status] || '待核查' }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>

        <!-- 勾稽违反详情 -->
        <div v-if="selectedRecord.record_type === 'violation' && selectedRecord.values" class="full-context">
          <el-divider />
          <div class="context-title">📋 违反详情</div>
          <el-table :data="Object.entries(selectedRecord.values).map(([k, v]) => ({ field: k, value: v }))" border size="small" max-height="200">
            <el-table-column prop="field" label="字段" min-width="150" />
            <el-table-column prop="value" label="值" min-width="150">
              <template #default="{ row }">
                {{ row.value !== undefined && row.value !== null ? row.value : 'null' }}
              </template>
            </el-table-column>
          </el-table>
          <div v-if="selectedRecord.diff !== undefined" style="margin-top: 8px; font-size: 13px; color: #f56c6c;">
            ⚠️ 偏差值: {{ selectedRecord.diff.toFixed(4) }}
          </div>
        </div>

        <div v-else-if="selectedRecord.features" class="full-context">
          <el-divider />
          <div class="context-title">📋 特征值</div>
          <pre>{{ JSON.stringify(selectedRecord.features, null, 2) }}</pre>
        </div>
      </div>
      <template #footer>
        <el-button @click="detailDialogVisible = false">关闭</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { useAnalysisStore } from '../stores/analysis'
import { scenariosApi } from '../api/scenarios'
import StartAnalysis from '../components/StartAnalysis.vue'
import AnalysisSummary from '../components/AnalysisSummary.vue'
import DataDecodeAndTrace from '../components/DataDecodeAndTrace.vue'

const router = useRouter()
const sessionStore = useSessionStore()
const analysisStore = useAnalysisStore()

// ===== 状态 =====
const loading = ref(true)
const error = ref('')
const activeTab = ref('config')

// 配置状态
const candidates = ref([])
const fieldMapping = ref({})
const mappingText = ref('')
const parsedMapping = ref({})
const parsedUnmatched = ref([])
const scenarioConfigSaved = ref(false)
const mappingConfigSaved = ref(false)
const savingScenarios = ref(false)
const savingMapping = ref(false)
const parsing = ref(false)

// 执行状态
const executing = ref(false)
const execProgress = ref(0)
const execMessage = ref('')

// 洞察结果
const scenarioResults = ref([])
const summary = ref({ total: 0, completed: 0, failed: 0 })
const expandedCard = ref(null)
const hasResults = ref(false)
const insights = ref({})
const loadingInsights = ref(false)

// 记录发现状态
const allRecords = ref([])
const totalRecords = ref(0)
const detailDialogVisible = ref(false)
const selectedRecord = ref(null)

// 数据追溯
const dataTraceRef = ref(null)
const traceContext = ref(null)

// ===== 计算属性 =====
const enabledCount = computed(() => candidates.value.filter(c => c.enabled).length)

const hasDbConfig = computed(() => {
  const session = sessionStore.currentSession
  if (!session) return false
  const tablesInfo = session.tables_info || {}
  return !!tablesInfo.db_config
})

const parquetData = computed(() => {
  // 从 analysisStore 获取原始数据
  const allTables = analysisStore.reportData?.all_tables
  if (!allTables) return []
  const merged = allTables.merged
  if (!merged) return []
  return merged.data || []
})

// ===== 生命周期 =====
onMounted(() => { loadData() })

// ===== 加载数据 =====
async function loadData() {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) { error.value = '请先加载项目'; loading.value = false; return }

  loading.value = true
  error.value = ''
  try {
    const response = await scenariosApi.get(sessionId)
    candidates.value = response.candidates || []
    scenarioConfigSaved.value = response.status === 'draft' || response.status === 'executed'
    if (response.field_mapping) fieldMapping.value = response.field_mapping
    if (response.results && response.results.length > 0) {
      scenarioResults.value = response.results
      hasResults.value = true
      buildRecordsFromResults(response.results)
    }
    if (response.insights) {
      insights.value = response.insights
    }
    if (response.status === 'draft') scenarioConfigSaved.value = true
    if (Object.keys(fieldMapping.value).length > 0) mappingConfigSaved.value = true
    updateSummary()
    console.log('[场景分析] 加载完成，记录数:', totalRecords.value)
  } catch (err) {
    console.error('[场景分析] 加载失败:', err)
    error.value = err.message || '加载失败'
    if (err.message && err.message.includes('JSON')) {
      candidates.value = []
      scenarioResults.value = []
      hasResults.value = false
    }
  } finally {
    loading.value = false
  }
}

function updateSummary() {
  const total = scenarioResults.value.length
  const completed = scenarioResults.value.filter(r => r.status === 'completed').length
  const failed = scenarioResults.value.filter(r => r.status === 'failed').length
  summary.value = { total, completed, failed }
}

function buildRecordsFromResults(results) {
  const records = []
  const typeDisplay = {
    'cluster': '聚类',
    'violation': '勾稽',
    'outlier': '异常',
    'missing': '缺失',
    'duplicate': '重复',
    'entity_concentration': '集中'
  }

  results.forEach(scenario => {
    if (scenario.status !== 'completed') return
    const scenarioName = scenario.business_name || scenario.name
    const scenarioId = String(scenario.scenario_id || '')
    const recs = scenario.records || []
    recs.forEach(r => {
      let recordType = r.record_type || 'other'
      let fieldDisplay = r.field_display || r.field || '—'
      let valueDisplay = r.value
      let valuesDisplay = null

      if (recordType === 'cluster') {
        fieldDisplay = '归属群组'
        valueDisplay = `群组 ${r.cluster_id}`
      } else if (recordType === 'entity_concentration') {
        fieldDisplay = r.entity || '实体'
        valueDisplay = r.entity || '—'
      } else if (recordType === 'violation') {
        fieldDisplay = r.rule || '勾稽规则'
        if (r.values && typeof r.values === 'object' && Object.keys(r.values).length > 0) {
          const pairs = Object.entries(r.values).map(([f, v]) => {
            const displayName = fieldMapping.value[f] || f
            const valStr = v !== undefined && v !== null ? (typeof v === 'number' ? v.toFixed(2) : String(v)) : 'null'
            return `${displayName}=${valStr}`
          })
          valuesDisplay = pairs.join(', ')
          valueDisplay = valuesDisplay
        } else if (r.fields && Array.isArray(r.fields) && r.fields.length > 0) {
          const displayNames = r.fields.map(f => fieldMapping.value[f] || f)
          valuesDisplay = displayNames.join(', ')
          valueDisplay = valuesDisplay
        } else {
          valueDisplay = '违反'
          valuesDisplay = '违反'
        }
      } else if (recordType === 'missing') {
        if (r.missing_fields && Array.isArray(r.missing_fields)) {
          valueDisplay = r.missing_fields.map(f => fieldMapping.value[f] || f).join(', ')
        }
      } else if (recordType === 'duplicate') {
        fieldDisplay = '重复记录'
        valueDisplay = '—'
      }

      records.push({
        ...r,
        scenario_id: scenarioId,
        scenario_name: scenarioName,
        record_type: recordType,
        record_type_display: typeDisplay[recordType] || '其他',
        field_display: fieldDisplay,
        value_display: valueDisplay,
        values_display: valuesDisplay
      })
    })
  })

  allRecords.value = records
  totalRecords.value = records.length
  console.log(`[记录发现] 总共 ${records.length} 条记录`)
}

// ===== 全选/全不选 =====
function selectAll(selected) {
  candidates.value.forEach(c => { c.enabled = selected })
}

// ===== 场景配置保存 =====
async function saveScenarioConfig() {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) { ElMessage.warning('请先加载项目'); return }
  savingScenarios.value = true
  try {
    await scenariosApi.update(sessionId, candidates.value)
    scenarioConfigSaved.value = true
    ElMessage.success('场景配置已保存')
  } catch (err) { ElMessage.error('保存失败: ' + err.message) }
  finally { savingScenarios.value = false }
}

function resetScenarioConfig() {
  candidates.value.forEach(c => c.enabled = c.default_enabled !== false)
  scenarioConfigSaved.value = false
  ElMessage.success('场景配置已重置')
}

// ===== 字段映射保存 =====
async function saveMappingConfig() {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) { ElMessage.warning('请先加载项目'); return }
  if (Object.keys(fieldMapping.value).length === 0) {
    ElMessage.warning('请先配置字段映射')
    return
  }
  savingMapping.value = true
  try {
    await scenariosApi.saveMapping(sessionId, fieldMapping.value)
    mappingConfigSaved.value = true
    ElMessage.success('字段映射已保存')
  } catch (err) { ElMessage.error('保存失败: ' + err.message) }
  finally { savingMapping.value = false }
}

// ===== 配置标签页功能 =====
function loadSampleMapping() {
  mappingText.value = 'CompanyFixAsset: 固定资产情况表\nDeclareDate: 公告日期\nCompanyCode: 公司代码\nCompanyName: 公司名称\nReportDate: 报表截止日期\nCurrency: 币种\nCunit: 单位\nReportRange: 合并范围\nCompanyFixAsset1: 固定资产名称代码\nCompanyFixAsset4: 账面原值_期初余额\nCompanyFixAsset17: 账面原值_期末余额\nCompanyFixAsset31: 累计折旧_期末余额\nCompanyFixAsset35: 账面净值_期末余额\nCompanyFixAsset49: 减值准备_期末余额\nCompanyFixAsset53: 账面价值_期末余额'
  ElMessage.success('已加载示例映射，请点击"解析映射"')
}

async function parseMapping() {
  if (!mappingText.value || !mappingText.value.trim()) { ElMessage.warning('请输入字段映射信息'); return }
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) { ElMessage.warning('请先加载项目'); return }

  parsing.value = true
  try {
    const response = await scenariosApi.parseMapping(sessionId, mappingText.value)
    parsedMapping.value = response.mapping || {}
    parsedUnmatched.value = response.unmatched || []
    if (Object.keys(parsedMapping.value).length === 0) ElMessage.warning('未识别到任何映射关系')
    else ElMessage.success(`成功解析 ${Object.keys(parsedMapping.value).length} 条映射`)
  } catch (err) { ElMessage.error('解析失败: ' + (err.message || '未知错误')) }
  finally { parsing.value = false }
}

function removeParsedMapping(key) { delete parsedMapping.value[key] }
function clearParsed() { parsedMapping.value = {}; parsedUnmatched.value = [] }
function confirmMapping() {
  fieldMapping.value = { ...fieldMapping.value, ...parsedMapping.value }
  ElMessage.success(`已确认 ${Object.keys(parsedMapping.value).length} 条映射`)
  parsedMapping.value = {}; parsedUnmatched.value = []
}
function removeMapping(key) { delete fieldMapping.value[key] }
function clearAllMapping() { fieldMapping.value = {}; ElMessage.success('已清空所有映射') }

// ===== 核心方法：执行场景 =====
async function runScenarioAnalysis() {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) { ElMessage.warning('请先加载项目'); return }

  if (!scenarioConfigSaved.value) {
    await saveScenarioConfig()
    if (!scenarioConfigSaved.value) {
      ElMessage.warning('场景配置未保存，请手动保存')
      return
    }
  }

  if (enabledCount.value === 0) {
    ElMessage.warning('请至少勾选一个场景')
    return
  }

  if (Object.keys(fieldMapping.value).length === 0) {
    ElMessage.warning('请先配置字段映射，然后重新执行')
    return
  }

  await saveMappingConfig()
  if (!mappingConfigSaved.value) {
    ElMessage.warning('字段映射未保存，请检查')
    return
  }

  executing.value = true
  execProgress.value = 0
  execMessage.value = '准备执行场景...'

  try {
    execProgress.value = 20
    execMessage.value = '正在执行场景分析...'
    const executeResponse = await scenariosApi.execute(sessionId)
    const results = executeResponse.results || []
    if (results.length === 0) {
      ElMessage.warning('执行完成，但没有返回结果')
      executing.value = false
      return
    }
    scenarioResults.value = results
    hasResults.value = true
    buildRecordsFromResults(results)
    updateSummary()
    ElMessage.success(`成功执行 ${executeResponse.count || results.length} 个场景`)

    execProgress.value = 70
    execMessage.value = '正在生成业务解读...'
    const translateResponse = await scenariosApi.translate(sessionId, fieldMapping.value)
    const translatedResults = translateResponse.results || []
    if (translatedResults.length > 0) {
      scenarioResults.value = translatedResults
      buildRecordsFromResults(translatedResults)
      ElMessage.success(`成功生成 ${translatedResults.length} 个场景的解读`)
    }

    execProgress.value = 90
    execMessage.value = '正在生成洞察分析...'
    await loadInsights(sessionId)

    execProgress.value = 100
    execMessage.value = '全部完成'

    setTimeout(() => {
      activeTab.value = 'dashboard'
    }, 500)

  } catch (err) {
    ElMessage.error('执行失败: ' + err.message)
    console.error('[场景分析] 执行失败:', err)
  } finally {
    executing.value = false
    setTimeout(() => {
      execProgress.value = 0
      execMessage.value = ''
    }, 2000)
  }
}

async function loadInsights(sessionId) {
  loadingInsights.value = true
  try {
    const response = await scenariosApi.getInsights(sessionId)
    if (response.has_insights) {
      insights.value = response.insights
      console.log('[场景分析] 洞察加载成功')
    }
  } catch (err) {
    console.error('[场景分析] 加载洞察失败:', err)
  } finally {
    loadingInsights.value = false
  }
}

function toggleCardDetail(id) { expandedCard.value = expandedCard.value === id ? null : id }

// ===== 跳转方法 =====
function goToConfig() { activeTab.value = 'config' }

function goToRecords(scenario) {
  activeTab.value = 'records'
  // 传递筛选条件
  const scenarioId = String(scenario.scenario_id || '')
  if (scenarioId && dataTraceRef.value) {
    dataTraceRef.value.updateFilter('scenario', scenarioId)
  }
}

function showRecordDetail(row) {
  selectedRecord.value = row
  detailDialogVisible.value = true
}

function updateRecordStatus(row, status) {
  row.status = status
  ElMessage.success(`状态已更新为 ${status}`)
}

// ===== 追溯方法 =====
function handleViewFullData(payload) {
  console.log('[追溯] 收到追溯请求:', payload)

  if (!payload || !payload.type) {
    ElMessage.warning('无效的追溯请求')
    return
  }

  let context = {}
  let description = ''

  switch (payload.type) {
    case 'rule': {
      if (payload.rule) {
        context = {
          rule: payload.rule,
          fields: payload.fields || []
        }
        description = `规则「${payload.rule.substring(0, 60)}...」的全量数据验证`
      } else {
        ElMessage.warning('规则信息不完整')
        return
      }
      break
    }
    case 'company': {
      context = { company_code: payload.data }
      description = `公司 ${payload.data} 全部数据`
      break
    }
    case 'field': {
      // 收集该字段的异常行号
      const rowIds = []
      for (const scenario of scenarioResults.value) {
        if (scenario.status !== 'completed') continue
        const records = scenario.records || []
        for (const record of records) {
          if (record.record_type === 'outlier' && record.field === payload.data) {
            if (record.row) rowIds.push(record.row)
          }
        }
      }
      if (rowIds.length > 0) {
        context = { row_ids: rowIds }
        description = `字段「${payload.data}」的异常记录（${rowIds.length} 条）`
      } else {
        ElMessage.warning('该字段没有关联的异常记录')
        return
      }
      break
    }
    case 'row': {
      context = { row_ids: [payload.data] }
      description = `行 ${payload.data} 的完整数据`
      break
    }
    case 'range': {
      context = { id_range: { start: payload.start, end: payload.end } }
      description = `数据段 ${payload.start}-${payload.end}`
      break
    }
    default: {
      ElMessage.warning('未知的追溯类型')
      return
    }
  }

  traceContext.value = context
  activeTab.value = 'records'

  // 切换到 SQL 模式
  if (dataTraceRef.value) {
    dataTraceRef.value.dataSource = 'sql'
    setTimeout(() => dataTraceRef.value.loadSqlData(), 300)
  }

  ElMessage.success(`已切换到数据追溯：${description}`)
}

function handleTrace(payload) {
  // 从数据解码组件触发的追溯
  if (payload.type === 'row' && payload.data) {
    const context = {
      row_ids: [payload.data]
    }
    traceContext.value = context
    // 数据解码组件会自己处理加载
  }
}

function goToUpload() { router.push('/upload') }

function onTabChange(tab) {
  if (tab.name === 'records') {
    // 切换到数据解码 tab 时，如果有追溯上下文则自动加载
    if (traceContext.value && dataTraceRef.value) {
      dataTraceRef.value.dataSource = 'sql'
      setTimeout(() => dataTraceRef.value.loadSqlData(), 300)
    }
  }
}
</script>

<style scoped>
.scenario-analysis {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}
.page-header {
  margin-bottom: 24px;
}
.page-header h2 {
  margin: 0 0 8px 0;
  color: #2c3e50;
}
.subtitle {
  color: #909399;
  margin: 0;
}
.loading-container {
  padding: 40px 0;
}
.error-container {
  padding: 60px 0;
}
.analysis-container {
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  padding: 20px;
}
.record-detail {
  padding: 4px 0;
}
.full-context {
  margin-top: 12px;
}
.context-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 8px;
}
.full-context pre {
  background: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  font-size: 12px;
  max-height: 300px;
  overflow: auto;
}

@media (max-width: 768px) {
  .scenario-analysis {
    padding: 12px;
  }
}
</style>