<template>
  <div class="scenario-dashboard">
    <div class="page-header">
      <h2>📊 场景仪表板</h2>
      <p class="subtitle">左侧查看技术结论，右侧生成业务解读</p>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="8" animated />
    </div>

    <!-- 错误/空状态 -->
    <div v-else-if="error" class="error-container">
      <el-result icon="error" :title="error" sub-title="请先执行场景推导">
        <template #extra>
          <el-button type="primary" @click="goToDerivation">去场景推导</el-button>
        </template>
      </el-result>
    </div>

    <!-- 主内容 -->
    <div v-else-if="hasResults" class="dashboard-content">
      <!-- 顶部概览 -->
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-value">{{ dataOverview.rows || 0 }}</div>
          <div class="stat-label">总行数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ dataOverview.columns || 0 }}</div>
          <div class="stat-label">总列数</div>
        </div>
        <div class="stat-card" :class="qualityClass">
          <div class="stat-value">{{ dataOverview.quality_score?.toFixed(1) || '--' }}</div>
          <div class="stat-label">质量评分</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ summary.total || 0 }}</div>
          <div class="stat-label">场景总数</div>
        </div>
      </div>

      <!-- 全局字段映射配置栏 -->
      <div class="mapping-bar">
        <div class="mapping-bar-left">
          <span class="mapping-bar-label">📝 字段映射</span>
          <el-tag v-if="Object.keys(fieldMapping).length > 0" size="small" type="success">
            已配置 {{ Object.keys(fieldMapping).length }} 个字段
          </el-tag>
          <el-tag v-else size="small" type="warning">未配置</el-tag>
        </div>
        <div class="mapping-bar-right">
          <el-button size="small" type="primary" plain @click="openGlobalMapping">
            {{ Object.keys(fieldMapping).length > 0 ? '✏️ 编辑映射' : '📝 配置映射' }}
          </el-button>
          <el-button
            size="small"
            type="primary"
            :loading="translating"
            :disabled="Object.keys(fieldMapping).length === 0"
            @click="translateAll"
          >
            {{ translating ? '翻译中...' : '🤖 生成解读' }}
          </el-button>
        </div>
      </div>

      <!-- 场景卡片列表 -->
      <div
        v-for="scenario in scenarios"
        :key="scenario.scenario_id"
        class="scenario-panel"
      >
        <!-- 场景标题 -->
        <div class="panel-header">
          <div class="header-left">
            <span class="scenario-status" :class="scenario.status">
              {{ scenario.status === 'completed' ? '✅' : '❌' }}
            </span>
            <span class="scenario-name">{{ scenario.business_name || scenario.name }}</span>
            <el-tag v-if="scenario.business_summary" size="small" type="success">
              💡 已解读
            </el-tag>
          </div>
          <div class="header-right">
            <el-button size="small" text @click="toggleExpand(scenario.scenario_id)">
              {{ expanded === scenario.scenario_id ? '收起' : '展开' }}
            </el-button>
          </div>
        </div>

        <!-- 左右栏主体 -->
        <div class="panel-body">
          <!-- 左侧：技术结论 -->
          <div class="panel-left">
            <div class="panel-label">📊 技术结论</div>
            <div class="tech-conclusions">
              <div
                v-for="(conclusion, idx) in scenario.conclusions"
                :key="idx"
                class="tech-item"
              >
                <span class="tech-icon">{{ getConclusionIcon(conclusion.type) }}</span>
                <span class="tech-text">{{ replaceFieldNames(conclusion.text) }}</span>
                <span v-if="conclusion.confidence" class="tech-confidence">
                  {{ (conclusion.confidence * 100).toFixed(0) }}%
                </span>
              </div>
            </div>
          </div>

          <!-- 右侧：业务解读 -->
          <div class="panel-right">
            <div class="panel-label">💡 业务解读</div>

            <div v-if="!scenario.business_summary" class="translate-placeholder">
              <div class="placeholder-text">
                <span>🔍 请先配置字段映射，然后点击"生成解读"</span>
              </div>
            </div>

            <div v-else class="business-content">
              <div class="business-summary">
                <span class="summary-icon">💡</span>
                <span class="summary-text">{{ scenario.business_summary }}</span>
              </div>

              <div v-if="scenario.business_findings && scenario.business_findings.length > 0" class="business-findings">
                <div class="findings-title">🔍 详细发现</div>
                <ul>
                  <li v-for="(item, idx) in scenario.business_findings" :key="idx">
                    {{ item }}
                  </li>
                </ul>
              </div>

              <div v-if="scenario.business_actions && scenario.business_actions.length > 0" class="business-actions">
                <div class="actions-title">📋 建议行动</div>
                <ul>
                  <li v-for="(item, idx) in scenario.business_actions" :key="idx">
                    {{ item }}
                  </li>
                </ul>
              </div>

              <div v-if="scenario._translation_error" class="business-error">
                <el-alert
                  :title="'翻译失败: ' + scenario._translation_error"
                  type="warning"
                  show-icon
                  :closable="false"
                />
              </div>
            </div>
          </div>
        </div>

        <!-- 展开详情 -->
        <div v-if="expanded === scenario.scenario_id" class="panel-detail">
          <el-divider />
          <div class="detail-content">
            <div class="detail-section">
              <div class="detail-title">📋 完整技术结论</div>
              <pre>{{ JSON.stringify(scenario.conclusions, null, 2) }}</pre>
            </div>
            <div v-if="scenario.business_summary" class="detail-section">
              <div class="detail-title">📋 完整业务解读</div>
              <pre>{{ JSON.stringify({ summary: scenario.business_summary, findings: scenario.business_findings, actions: scenario.business_actions }, null, 2) }}</pre>
            </div>
          </div>
        </div>
      </div>

      <!-- 底部操作栏 -->
      <div class="actions-bar">
        <el-button size="small" @click="loadData">🔄 刷新</el-button>
        <el-button size="small" @click="goToDerivation">🔧 场景管理</el-button>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-else-if="!loading" class="empty-state">
      <el-empty description="暂无场景结果，请先执行场景推导">
        <el-button type="primary" @click="goToDerivation">去场景推导</el-button>
      </el-empty>
    </div>

    <!-- 全局字段映射弹窗 -->
    <el-dialog
      v-model="mappingDialogVisible"
      title="📝 字段映射配置（全局）"
      width="750px"
      destroy-on-close
    >
      <div class="mapping-dialog">
        <el-alert
          title="输入字段名和中文名的对应关系，支持任意格式"
          type="info"
          show-icon
          :closable="false"
          style="margin-bottom: 16px"
        >
          <template #default>
            <div style="font-size: 12px; color: #909399; margin-top: 4px;">
              示例：companyfixasset1: 总资产, companyfixasset4: 净资产
              <br>或：companyfixasset1是总资产，companyfixasset4是净资产
              <br>也支持表格文本，系统会自动解析
            </div>
          </template>
        </el-alert>

        <el-input
          v-model="mappingText"
          type="textarea"
          :rows="6"
          placeholder="粘贴字段映射信息，支持任意格式..."
        />

        <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
          <el-button size="small" type="primary" @click="parseMapping" :loading="parsing">
            {{ parsing ? '解析中...' : '🔍 解析映射' }}
          </el-button>
          <el-button size="small" @click="mappingText = ''">清空</el-button>
          <el-button size="small" type="success" plain @click="loadSampleMapping">
            加载示例
          </el-button>
        </div>

        <!-- 解析结果 -->
        <div v-if="Object.keys(parsedMapping).length > 0 || parsedUnmatched.length > 0" class="mapping-result">
          <el-divider />
          <div class="result-title">解析结果</div>

          <div v-if="Object.keys(parsedMapping).length > 0" class="mapping-list">
            <div
              v-for="(value, key) in parsedMapping"
              :key="key"
              class="mapping-item"
            >
              <span class="mapping-key">{{ key }}</span>
              <span class="mapping-arrow">→</span>
              <el-input
                :model-value="value"
                size="small"
                style="width: 160px;"
                @update:model-value="(val) => { parsedMapping[key] = val }"
              />
              <el-button size="small" text type="danger" @click="removeMapping(key)">
                ✕
              </el-button>
            </div>
          </div>

          <div v-if="parsedUnmatched.length > 0" class="unmatched-list">
            <div class="unmatched-title">⚠️ 未识别字段：</div>
            <el-tag
              v-for="item in parsedUnmatched"
              :key="item"
              size="small"
              type="warning"
              style="margin: 2px;"
            >
              {{ item }}
            </el-tag>
          </div>

          <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
            <el-button
              size="small"
              type="primary"
              :disabled="Object.keys(parsedMapping).length === 0"
              @click="confirmMapping"
            >
              ✅ 确认映射
            </el-button>
            <el-button size="small" @click="clearParsed">清空结果</el-button>
          </div>
        </div>

        <!-- 当前已保存映射 -->
        <div v-if="Object.keys(currentFieldMapping).length > 0" class="current-mapping">
          <el-divider />
          <div class="result-title">当前已保存的映射（{{ Object.keys(currentFieldMapping).length }} 个）</div>
          <div class="mapping-list">
            <div
              v-for="(value, key) in currentFieldMapping"
              :key="key"
              class="mapping-item"
            >
              <span class="mapping-key">{{ key }}</span>
              <span class="mapping-arrow">→</span>
              <span class="mapping-value">{{ value }}</span>
              <el-button size="small" text type="danger" @click="removeCurrentMapping(key)">
                ✕
              </el-button>
            </div>
          </div>
          <el-button size="small" type="danger" plain @click="clearAllMapping" style="margin-top: 8px;">
            清空所有映射
          </el-button>
        </div>
      </div>

      <template #footer>
        <el-button @click="mappingDialogVisible = false">关闭</el-button>
        <el-button
          type="primary"
          @click="saveAndTranslate"
          :loading="translating"
          :disabled="Object.keys(currentFieldMapping).length === 0"
        >
          {{ translating ? '生成中...' : '✅ 确认并生成解读' }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { scenariosApi } from '../api/scenarios'

const router = useRouter()
const sessionStore = useSessionStore()

// ===== 状态 =====
const loading = ref(false)
const translating = ref(false)
const parsing = ref(false)
const error = ref('')
const scenarios = ref([])
const dataOverview = ref({ rows: 0, columns: 0, quality_score: 0 })
const summary = ref({ total: 0, completed: 0, failed: 0 })
const expanded = ref(null)

// ===== 字段映射（全局） =====
const fieldMapping = ref({})
const currentFieldMapping = ref({})

// ===== 弹窗状态 =====
const mappingDialogVisible = ref(false)
const mappingText = ref('')
const parsedMapping = ref({})
const parsedUnmatched = ref([])

// ===== 计算属性 =====
const hasResults = computed(() => scenarios.value.length > 0)

const qualityClass = computed(() => {
  const score = dataOverview.value.quality_score || 0
  if (score >= 80) return 'status-excellent'
  if (score >= 60) return 'status-good'
  return 'status-poor'
})

// ===== 生命周期 =====
onMounted(() => {
  loadData()
})

// ===== 加载数据 =====
async function loadData() {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    error.value = '请先加载项目'
    return
  }

  loading.value = true
  error.value = ''
  try {
    const response = await scenariosApi.getDashboard(sessionId)
    if (response.has_results) {
      scenarios.value = response.scenarios || []
      dataOverview.value = response.data_overview || { rows: 0, columns: 0, quality_score: 0 }
      summary.value = response.summary || { total: 0, completed: 0, failed: 0 }
      fieldMapping.value = response.field_mapping || {}
      currentFieldMapping.value = { ...fieldMapping.value }
    } else {
      scenarios.value = []
      error.value = response.message || '请先执行场景推导'
    }
  } catch (err) {
    error.value = err.message || '加载失败'
  } finally {
    loading.value = false
  }
}

// ===== 展开/收起 =====
function toggleExpand(id) {
  expanded.value = expanded.value === id ? null : id
}

// ===== 图标映射 =====
function getConclusionIcon(type) {
  const map = {
    'summary': '📌',
    'detail': '📋',
    'alert': '⚠️',
    'success': '✅',
    'error': '❌'
  }
  return map[type] || '📌'
}

// ===== 替换字段名为中文 =====
function replaceFieldNames(text) {
  if (!text) return text
  let result = text
  for (const [key, value] of Object.entries(fieldMapping.value)) {
    result = result.replace(new RegExp(key, 'g'), value)
  }
  return result
}

// ===== 打开全局映射弹窗 =====
function openGlobalMapping() {
  currentFieldMapping.value = { ...fieldMapping.value }
  parsedMapping.value = {}
  parsedUnmatched.value = []
  mappingText.value = ''
  mappingDialogVisible.value = true
}

// ===== 加载示例映射 =====
function loadSampleMapping() {
  mappingText.value =
    'CompanyFixAsset: 固定资产情况表\n' +
    'DeclareDate: 公告日期\n' +
    'CompanyCode: 公司代码\n' +
    'CompanyName: 公司名称\n' +
    'ReportDate: 报表截止日期\n' +
    'Currency: 币种\n' +
    'Cunit: 单位\n' +
    'ReportRange: 合并范围\n' +
    'CompanyFixAsset1: 固定资产名称代码\n' +
    'CompanyFixAsset2: 固定资产标准名称\n' +
    'CompanyFixAsset3: 固定资产披露名称\n' +
    'CompanyFixAsset4: 账面原值_期初余额\n' +
    'CompanyFixAsset17: 账面原值_期末余额\n' +
    'CompanyFixAsset31: 累计折旧_期末余额\n' +
    'CompanyFixAsset35: 账面净值_期末余额\n' +
    'CompanyFixAsset49: 减值准备_期末余额\n' +
    'CompanyFixAsset53: 账面价值_期末余额\n' +
    'AnnSource: 来源公告\n' +
    'Guid: Guid\n' +
    'IsDel: 是否删除\n' +
    'EntryDate: 录入日期\n' +
    'EntryTime: 录入时间'
  ElMessage.success('已加载示例映射，请点击"解析映射"')
}

// ===== 解析映射 =====
async function parseMapping() {
  if (!mappingText.value || !mappingText.value.trim()) {
    ElMessage.warning('请输入字段映射信息')
    return
  }

  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  parsing.value = true
  try {
    const response = await scenariosApi.parseMapping(sessionId, mappingText.value)
    parsedMapping.value = response.mapping || {}
    parsedUnmatched.value = response.unmatched || []

    if (Object.keys(parsedMapping.value).length === 0) {
      ElMessage.warning('未识别到任何映射关系，请检查输入格式')
    } else {
      ElMessage.success(`成功解析 ${Object.keys(parsedMapping.value).length} 条映射`)
    }
  } catch (err) {
    ElMessage.error('解析失败: ' + (err.message || '未知错误'))
    console.error('解析映射错误:', err)
  } finally {
    parsing.value = false
  }
}

function removeMapping(key) {
  delete parsedMapping.value[key]
}

function removeCurrentMapping(key) {
  delete currentFieldMapping.value[key]
  delete fieldMapping.value[key]
}

function clearParsed() {
  parsedMapping.value = {}
  parsedUnmatched.value = []
}

function clearAllMapping() {
  currentFieldMapping.value = {}
  fieldMapping.value = {}
  ElMessage.success('已清空所有映射')
}

function confirmMapping() {
  // 合并映射（保留已有，覆盖冲突）
  currentFieldMapping.value = { ...currentFieldMapping.value, ...parsedMapping.value }
  ElMessage.success(`已确认 ${Object.keys(parsedMapping.value).length} 条映射`)
  parsedMapping.value = {}
  parsedUnmatched.value = []
}

// ===== 保存并翻译 =====
async function saveAndTranslate() {
  // 先确认当前解析结果
  if (Object.keys(parsedMapping.value).length > 0) {
    await confirmMapping()
  }

  // 保存映射到全局
  fieldMapping.value = { ...currentFieldMapping.value }

  // 检查是否有映射
  if (Object.keys(fieldMapping.value).length === 0) {
    ElMessage.warning('请至少配置一条字段映射')
    return
  }

  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  translating.value = true
  try {
    const response = await scenariosApi.translate(sessionId, fieldMapping.value)
    scenarios.value = response.results || []
    mappingDialogVisible.value = false
    ElMessage.success(`成功翻译 ${scenarios.value.length} 个场景`)
  } catch (err) {
    ElMessage.error('翻译失败: ' + (err.message || '未知错误'))
  } finally {
    translating.value = false
  }
}

// ===== 批量翻译 =====
async function translateAll() {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  if (Object.keys(fieldMapping.value).length === 0) {
    openGlobalMapping()
    return
  }

  translating.value = true
  try {
    const response = await scenariosApi.translate(sessionId, fieldMapping.value)
    scenarios.value = response.results || []
    ElMessage.success(`成功翻译 ${scenarios.value.length} 个场景`)
  } catch (err) {
    ElMessage.error('翻译失败: ' + (err.message || '未知错误'))
  } finally {
    translating.value = false
  }
}

function goToDerivation() {
  router.push('/scenario-derivation')
}
</script>

<style scoped>
/* 样式与之前相同，新增 mapping-bar 样式 */

.scenario-dashboard {
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
.empty-state {
  padding: 60px 0;
}

/* ===== 统计卡片 ===== */
.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}
.stat-card {
  background: #f5f7fa;
  border-radius: 12px;
  padding: 20px;
  text-align: center;
}
.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #2c3e50;
}
.stat-label {
  font-size: 13px;
  color: #909399;
  margin-top: 4px;
}
.status-excellent .stat-value { color: #67c23a; }
.status-good .stat-value { color: #e6a23c; }
.status-poor .stat-value { color: #f56c6c; }

/* ===== 全局映射栏 ===== */
.mapping-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  background: #f0f7ff;
  border-radius: 8px;
  border: 1px solid #d9ecff;
  margin-bottom: 20px;
  flex-wrap: wrap;
  gap: 8px;
}
.mapping-bar-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.mapping-bar-label {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
}
.mapping-bar-right {
  display: flex;
  gap: 8px;
}

/* ===== 场景面板 ===== */
.scenario-panel {
  background: #fff;
  border-radius: 12px;
  border: 1px solid #e4e7ed;
  margin-bottom: 16px;
  overflow: hidden;
  transition: box-shadow 0.2s;
}
.scenario-panel:hover {
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  background: #f8f9fa;
  border-bottom: 1px solid #e4e7ed;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.scenario-status {
  font-size: 16px;
}
.scenario-name {
  font-size: 15px;
  font-weight: 600;
  color: #2c3e50;
}
.header-right {
  display: flex;
  gap: 4px;
}

/* ===== 左右栏 ===== */
.panel-body {
  display: flex;
  min-height: 200px;
}
.panel-left {
  flex: 1;
  padding: 16px 20px;
  border-right: 1px solid #e4e7ed;
  background: #fafafa;
}
.panel-right {
  flex: 1;
  padding: 16px 20px;
  background: #fff;
}

.panel-label {
  font-size: 12px;
  font-weight: 600;
  color: #909399;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 12px;
}

/* ===== 左侧技术结论 ===== */
.tech-conclusions {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.tech-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 4px 0;
  font-size: 13px;
  color: #555;
  line-height: 1.5;
}
.tech-icon {
  font-size: 14px;
  flex-shrink: 0;
  margin-top: 1px;
}
.tech-text {
  flex: 1;
  word-break: break-word;
}
.tech-confidence {
  font-size: 11px;
  color: #909399;
  flex-shrink: 0;
}

/* ===== 右侧业务解读 ===== */
.translate-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 140px;
}
.placeholder-text span {
  font-size: 14px;
  color: #909399;
}

.business-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.business-summary {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 14px;
  background: #f0f9ff;
  border-radius: 8px;
  border-left: 4px solid #409eff;
}
.summary-icon {
  font-size: 18px;
  flex-shrink: 0;
}
.summary-text {
  font-size: 15px;
  font-weight: 500;
  color: #2c3e50;
  line-height: 1.5;
}

.business-findings, .business-actions {
  padding: 0 4px;
}
.findings-title, .actions-title {
  font-size: 13px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 6px;
}
.business-findings ul, .business-actions ul {
  margin: 0;
  padding-left: 20px;
}
.business-findings li, .business-actions li {
  font-size: 13px;
  color: #555;
  line-height: 1.6;
  padding: 2px 0;
}

.business-error {
  margin-top: 8px;
}

/* ===== 展开详情 ===== */
.panel-detail {
  padding: 16px 20px;
  background: #f8f9fa;
}
.detail-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.detail-section {
  background: #fff;
  border-radius: 8px;
  padding: 12px;
  border: 1px solid #e4e7ed;
}
.detail-title {
  font-size: 12px;
  font-weight: 600;
  color: #909399;
  margin-bottom: 8px;
}
.detail-section pre {
  margin: 0;
  font-size: 12px;
  font-family: 'Consolas', 'Courier New', monospace;
  white-space: pre-wrap;
  word-break: break-all;
  background: #f5f7fa;
  padding: 8px;
  border-radius: 4px;
  max-height: 300px;
  overflow-y: auto;
}

/* ===== 底部操作栏 ===== */
.actions-bar {
  display: flex;
  gap: 12px;
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #e4e7ed;
}

/* ===== 映射弹窗 ===== */
.mapping-dialog {
  max-height: 500px;
  overflow-y: auto;
}
.mapping-result {
  margin-top: 12px;
}
.result-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 8px;
}
.mapping-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.mapping-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  background: #f5f7fa;
  border-radius: 6px;
  flex-wrap: wrap;
}
.mapping-key {
  font-family: 'Consolas', monospace;
  font-size: 13px;
  font-weight: 500;
  color: #2c3e50;
  min-width: 100px;
}
.mapping-arrow {
  color: #909399;
}
.mapping-value {
  font-size: 13px;
  color: #409eff;
}
.unmatched-list {
  margin-top: 8px;
}
.unmatched-title {
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
}
.current-mapping {
  margin-top: 12px;
}

/* ===== 响应式 ===== */
@media (max-width: 992px) {
  .panel-body {
    flex-direction: column;
  }
  .panel-left {
    border-right: none;
    border-bottom: 1px solid #e4e7ed;
  }
  .detail-content {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .stats-row {
    grid-template-columns: repeat(3, 1fr);
  }
  .panel-header {
    flex-wrap: wrap;
    gap: 8px;
  }
  .mapping-bar {
    flex-direction: column;
    align-items: stretch;
  }
  .mapping-bar-right {
    justify-content: flex-end;
  }
}
</style>