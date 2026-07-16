<template>
  <div class="upload-page">
    <el-steps :active="currentStep" finish-status="success" align-center style="margin-bottom: 30px;">
      <el-step title="选择数据源" />
      <el-step title="数据预览" />
      <el-step title="开始分析" />
    </el-steps>

    <!-- ==================== 数据源类型切换 ==================== -->
    <el-radio-group v-model="dataSourceType" size="large" style="margin-bottom: 20px;" @change="onDataSourceChange">
      <el-radio-button value="file">📁 文件上传</el-radio-button>
      <el-radio-button value="database">🗄️ 数据库</el-radio-button>
      <el-radio-button value="demo">📊 示例数据</el-radio-button>
    </el-radio-group>

    <!-- ============================================================ -->
    <!-- 文件上传 -->
    <!-- ============================================================ -->
    <div v-if="dataSourceType === 'file'" class="upload-area">
      <div v-if="!uploadData" class="drop-zone" @dragover.prevent @drop.prevent="handleDrop">
        <el-upload
          ref="uploadRef"
          drag
          :auto-upload="false"
          :on-change="handleFileChange"
          :on-remove="handleRemove"
          :limit="1"
          action="#"
        >
          <el-icon class="el-icon--upload"><Upload /></el-icon>
          <div class="el-upload__text">
            拖拽文件到此处，或 <em>点击选择</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">
              支持 CSV, Excel, JSON, TXT 格式，文件大小不超过 100MB
            </div>
          </template>
        </el-upload>
      </div>

      <!-- 文件预览 -->
      <div v-else class="file-preview">
        <div class="file-info">
          <div class="file-header">
            <el-icon><Document /></el-icon>
            <span class="file-name">{{ uploadData.file_name }}</span>
            <el-tag size="small" type="success">已上传</el-tag>
            <el-button size="small" type="danger" plain @click="handleReupload">重新选择</el-button>
          </div>
          <div class="file-stats">
            <span>📊 {{ uploadData.rows }} 行</span>
            <span>📋 {{ uploadData.columns }} 列</span>
          </div>
        </div>

        <!-- 字段类型调整 -->
        <div class="field-types">
          <h4>📋 字段类型调整 <el-tag size="small" type="info">点击下拉修改</el-tag></h4>
          <div class="type-grid">
            <div v-for="(type, field) in editableTypes" :key="field" class="type-item">
              <span class="field-name">{{ field }}</span>
              <el-select
                v-model="editableTypes[field]"
                size="small"
                style="width: 120px;"
                @change="onTypeChange(field, $event)"
              >
                <el-option label="连续变量" value="continuous" />
                <el-option label="分类变量" value="categorical" />
                <el-option label="数值型分类" value="categorical_numeric" />
                <el-option label="有序分类" value="ordinal" />
                <el-option label="日期时间" value="datetime" />
                <el-option label="标识符" value="identifier" />
                <el-option label="文本" value="text" />
                <el-option label="排除" value="exclude" />
              </el-select>
            </div>
          </div>
        </div>

        <!-- 数据预览 -->
        <div class="data-preview">
          <h4>数据预览（前 100 行）</h4>
          <el-table :data="uploadData.preview.head" border size="small" max-height="300">
            <el-table-column
              v-for="col in uploadData.preview.columns"
              :key="col"
              :prop="col"
              :label="col"
              width="120"
              show-overflow-tooltip
            />
          </el-table>
        </div>

        <!-- 操作按钮 -->
        <div class="actions">
          <el-button type="primary" :loading="analyzing" @click="handleStartAnalysis">
            {{ analyzing ? '分析中...' : '🚀 开始分析' }}
          </el-button>
        </div>

        <!-- 分析进度 -->
        <div v-if="analyzing" class="progress-section">
          <el-progress :percentage="progress" :format="formatProgress" />
          <p class="status-message">{{ statusMessage }}</p>
        </div>
      </div>
    </div>

    <!-- ============================================================ -->
    <!-- 数据库连接 -->
    <!-- ============================================================ -->
    <div v-if="dataSourceType === 'database'" class="upload-area">
      <el-alert
        title="连接 SQL Server 数据库，支持多表加载"
        type="info"
        show-icon
        :closable="false"
        style="margin-bottom: 16px"
      />

      <el-form :model="dbForm" label-width="120px" style="max-width: 600px;">
        <el-form-item label="数据库配置">
          <el-select
            v-model="dbForm.configName"
            placeholder="选择已有配置"
            style="width: 100%"
            @change="onDbConfigChange"
          >
            <el-option
              v-for="cfg in dbConfigs"
              :key="cfg.name"
              :label="cfg.name"
              :value="cfg.name"
            />
          </el-select>
        </el-form-item>

        <!-- ✅ 改为文本域，支持多表输入 -->
        <el-form-item label="表名">
          <el-input
            v-model="dbForm.tableNamesInput"
            type="textarea"
            :rows="3"
            placeholder="输入表名，支持逗号、空格、换行分隔&#10;例如：users, orders, products"
            @input="parseTableNames"
          />
          <div v-if="parsedTableNames.length > 0" class="table-tags">
            <el-tag
              v-for="name in parsedTableNames"
              :key="name"
              size="small"
              type="info"
              style="margin: 2px;"
            >
              {{ name }}
            </el-tag>
            <span style="font-size: 12px; color: #909399; margin-left: 8px;">
              共 {{ parsedTableNames.length }} 个表
            </span>
          </div>
        </el-form-item>

        <el-form-item label="加载行数">
          <el-input-number v-model="dbForm.limit" :min="100" :max="100000" :step="1000" />
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            :loading="loadingDb"
            :disabled="!dbForm.configName || parsedTableNames.length === 0"
            @click="handleLoadDbTables"
          >
            {{ loadingDb ? '加载中...' : '🔌 加载表' }}
          </el-button>
        </el-form-item>
      </el-form>

      <!-- 加载进度 -->
      <div v-if="loadingDb" class="loading-progress">
        <el-progress :percentage="dbLoadProgress" />
        <p class="status-message">{{ dbLoadStatus }}</p>
      </div>

      <!-- ✅ 关系确认 -->
      <TableRelationConfirm
        v-if="dbData && candidateRelations.length > 0 && !relationsConfirmed"
        :relations="candidateRelations"
        :table-names="dbTableNames"
        @confirm="handleRelationsConfirmed"
        @skip="handleRelationsSkipped"
      />

      <!-- 数据库加载结果 -->
      <div v-if="dbData" class="file-preview">
        <div class="file-info">
          <div class="file-header">
            <el-icon><DataBoard /></el-icon>
            <span class="file-name">已加载 {{ dbTableNames.length }} 个表</span>
            <el-tag size="small" type="success">已加载</el-tag>
          </div>
          <div class="file-stats">
            <span>📊 总表数: {{ dbTableNames.length }}</span>
            <span>📋 总列数: {{ totalColumns }}</span>
            <span>📈 总行数: {{ totalRows }}</span>
          </div>
        </div>

        <!-- 表切换 -->
        <div class="table-switcher">
          <el-radio-group v-model="selectedTable" size="small" @change="onTableChange">
            <el-radio-button
              v-for="name in dbTableNames"
              :key="name"
              :label="name"
            />
          </el-radio-group>
        </div>

        <!-- 当前表的信息 -->
        <div v-if="currentTableData" class="table-info">
          <el-descriptions :column="3" border size="small">
            <el-descriptions-item label="表名">{{ selectedTable }}</el-descriptions-item>
            <el-descriptions-item label="行数">{{ currentTableData.rows }}</el-descriptions-item>
            <el-descriptions-item label="列数">{{ currentTableData.columns }}</el-descriptions-item>
          </el-descriptions>
        </div>

        <!-- 字段类型调整 -->
        <div class="field-types">
          <h4>📋 字段类型调整 <el-tag size="small" type="info">点击下拉修改</el-tag></h4>
          <div class="type-grid">
            <div v-for="(type, field) in currentEditableTypes" :key="field" class="type-item">
              <span class="field-name">{{ field }}</span>
              <el-select
                v-model="currentEditableTypes[field]"
                size="small"
                style="width: 120px;"
                @change="onDbTypeChange(field, $event)"
              >
                <el-option label="连续变量" value="continuous" />
                <el-option label="分类变量" value="categorical" />
                <el-option label="数值型分类" value="categorical_numeric" />
                <el-option label="有序分类" value="ordinal" />
                <el-option label="日期时间" value="datetime" />
                <el-option label="标识符" value="identifier" />
                <el-option label="文本" value="text" />
                <el-option label="排除" value="exclude" />
              </el-select>
            </div>
          </div>
        </div>

        <!-- 数据预览 -->
        <div class="data-preview">
          <h4>数据预览（前 100 行）</h4>
          <el-table :data="currentPreviewData" border size="small" max-height="300">
            <el-table-column
              v-for="col in currentPreviewColumns"
              :key="col"
              :prop="col"
              :label="col"
              width="120"
              show-overflow-tooltip
            />
          </el-table>
        </div>

        <!-- 操作按钮 -->
        <div class="actions">
          <el-button type="primary" :loading="analyzing" @click="handleStartDbAnalysis">
            {{ analyzing ? '分析中...' : '🚀 开始分析' }}
          </el-button>
        </div>

        <!-- 分析进度 -->
        <div v-if="analyzing" class="progress-section">
          <el-progress :percentage="progress" :format="formatProgress" />
          <p class="status-message">{{ statusMessage }}</p>
        </div>
      </div>
    </div>

    <!-- ============================================================ -->
    <!-- 示例数据 -->
    <!-- ============================================================ -->
    <div v-if="dataSourceType === 'demo'" class="upload-area">
      <el-alert
        title="选择示例数据集，快速体验 AutoStat 功能"
        type="info"
        show-icon
        :closable="false"
        style="margin-bottom: 20px"
      />

      <div class="demo-grid">
        <el-card
          v-for="demo in demoDatasets"
          :key="demo.key"
          class="demo-card"
          shadow="hover"
          @click="handleLoadDemo(demo.key)"
        >
          <div class="demo-icon">{{ demo.icon }}</div>
          <div class="demo-name">{{ demo.name }}</div>
          <div class="demo-desc">{{ demo.description }}</div>
          <div class="demo-meta">{{ demo.rows }} 行 × {{ demo.cols }} 列</div>
          <el-button type="primary" size="small" style="margin-top: 12px;">加载此数据</el-button>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { useAnalysisStore } from '../stores/analysis'
import { dataApi } from '../api/data'
import { configApi } from '../api/config'
import TableRelationConfirm from '../components/TableRelationConfirm.vue'

const router = useRouter()
const sessionStore = useSessionStore()
const analysisStore = useAnalysisStore()

// ==================== 状态 ====================
const dataSourceType = ref('file')
const currentStep = ref(0)
const analyzing = ref(false)
const progress = ref(0)
const statusMessage = ref('')
const uploadData = ref(null)
const selectedFile = ref(null)
const uploadRef = ref(null)

// 字段类型
const editableTypes = ref({})
const originalTypes = ref({})

// 数据库
const dbConfigs = ref([])
const dbForm = reactive({
  configName: '',
  tableNamesInput: '',
  limit: 5000
})
const parsedTableNames = ref([])
const loadingDb = ref(false)
const dbData = ref(null)
const dbEditableTypes = ref({})
const dbOriginalTypes = ref({})
const candidateRelations = ref([])
const dbTableNames = ref([])
const selectedTable = ref('')
const currentTableData = ref(null)
const currentEditableTypes = ref({})
const currentPreviewData = ref([])
const currentPreviewColumns = ref([])
const dbLoadProgress = ref(0)
const dbLoadStatus = ref('')
const relationsConfirmed = ref(false)
const confirmedRelations = ref([])
const dbLoadResults = ref({})

// 示例数据
const demoDatasets = [
  { key: 'sales', name: '销售数据', icon: '📊', description: '零售销售分析', rows: 5000, cols: 9 },
  { key: 'user', name: '用户数据', icon: '👥', description: '用户行为分析', rows: 3000, cols: 10 },
  { key: 'medical', name: '医疗数据', icon: '🏥', description: '患者就诊分析', rows: 2000, cols: 11 }
]

// 当前有效的 session_id
const currentSessionId = ref(null)

onMounted(async () => {
  await loadDbConfigs()
})

async function loadDbConfigs() {
  try {
    const result = await configApi.getDatabase()
    dbConfigs.value = result || []
  } catch (err) {
    console.error('加载数据库配置失败:', err)
  }
}

function onDataSourceChange() {
  // 切换数据源时重置状态
}

// ==================== 文件上传 ====================
function handleFileChange(file) {
  selectedFile.value = file.raw
  handleUpload()
}

function handleRemove() {
  selectedFile.value = null
  uploadData.value = null
}

function handleDrop(e) {
  const files = e.dataTransfer.files
  if (files.length > 0) {
    selectedFile.value = files[0]
    handleUpload()
  }
}

function handleReupload() {
  uploadData.value = null
  selectedFile.value = null
  currentStep.value = 0
  if (uploadRef.value) {
    uploadRef.value.clearFiles()
  }
}

async function handleUpload() {
  if (!selectedFile.value) return

  try {
    if (!sessionStore.currentSessionId) {
      await sessionStore.createSession(selectedFile.value.name)
    }

    const result = await sessionStore.uploadFile(selectedFile.value)
    uploadData.value = result
    currentStep.value = 1

    editableTypes.value = { ...result.variable_types }
    originalTypes.value = { ...result.variable_types }

    ElMessage.success('文件上传成功')
  } catch (err) {
    ElMessage.error('上传失败: ' + err.message)
  }
}

function onTypeChange(field, newType) {
  console.log(`字段 ${field} 类型变更为: ${newType}`)
}

// ==================== 数据库连接 ====================
function onDbConfigChange() {}

// ==================== 多表解析 ====================
function parseTableNames() {
  const input = dbForm.tableNamesInput || ''
  const parts = input.split(/[,，\s\n]+/).map(s => s.trim())
  parsedTableNames.value = parts.filter(s => s.length > 0)
}

// ==================== 加载多表 ====================
async function handleLoadDbTables() {
  if (!dbForm.configName || parsedTableNames.value.length === 0) {
    ElMessage.warning('请选择配置并输入至少一个表名')
    return
  }

  loadingDb.value = true
  dbData.value = null
  dbLoadProgress.value = 0
  dbLoadStatus.value = '正在加载表...'
  relationsConfirmed.value = false

  try {
    const config = dbConfigs.value.find(c => c.name === dbForm.configName)
    if (!config) {
      ElMessage.error('配置不存在')
      return
    }

    const result = await dataApi.loadDatabase({
      config: config,
      table_names: parsedTableNames.value,
      limit: dbForm.limit
    })

    dbLoadProgress.value = 100
    dbLoadStatus.value = '加载完成'

    if (result && result.tables) {
      dbData.value = result
      dbTableNames.value = Object.keys(result.tables)

      // 保存候选关系
      candidateRelations.value = result.candidate_relations || []

      // 选择第一个表
      if (dbTableNames.value.length > 0) {
        selectedTable.value = dbTableNames.value[0]
        updateCurrentTable(selectedTable.value)
      }

      if (result.session_id) {
        sessionStore.currentSessionId = result.session_id
        localStorage.setItem('lastSessionId', result.session_id)
        await sessionStore.loadSession(result.session_id)
        // 保存关系到 store
        sessionStore.relationships = candidateRelations.value
      }

      ElMessage.success(`成功加载 ${dbTableNames.value.length} 个表`)
    }
  } catch (err) {
    ElMessage.error('加载失败: ' + err.message)
  } finally {
    loadingDb.value = false
  }
}

function updateCurrentTable(tableName) {
  if (!dbData.value || !dbData.value.tables[tableName]) return

  const table = dbData.value.tables[tableName]
  currentTableData.value = {
    rows: table.rows,
    columns: table.columns
  }
  currentEditableTypes.value = { ...table.variable_types }
  currentPreviewData.value = table.preview.head || []
  currentPreviewColumns.value = table.preview.columns || []
}

function onTableChange(value) {
  updateCurrentTable(value)
}

function onDbTypeChange(field, newType) {
  console.log(`数据库字段 ${field} 类型变更为: ${newType}`)
}

// ==================== 关系确认 ====================
function handleRelationsConfirmed(relations) {
  confirmedRelations.value = relations
  relationsConfirmed.value = true
  ElMessage.success(`已确认 ${relations.length} 条表间关系`)
}

function handleRelationsSkipped() {
  relationsConfirmed.value = true
  ElMessage.info('已跳过关系配置')
}

// ==================== 示例数据 ====================
async function handleLoadDemo(datasetKey) {
  try {
    const result = await dataApi.loadDemo(datasetKey)
    if (result && result.session_id) {
      currentSessionId.value = result.session_id
      sessionStore.currentSessionId = result.session_id
      localStorage.setItem('lastSessionId', result.session_id)

      sessionStore.currentSession = {
        source_name: result.source_name,
        data_shape: { rows: result.rows, columns: result.columns },
        variable_types: result.variable_types
      }

      uploadData.value = {
        file_name: `${datasetKey}_demo.csv`,
        rows: result.rows,
        columns: result.columns,
        variable_types: result.variable_types,
        preview: result.preview
      }

      editableTypes.value = { ...result.variable_types }
      originalTypes.value = { ...result.variable_types }
      currentStep.value = 1

      ElMessage.success(`已加载 ${result.source_name}`)

      dataSourceType.value = 'file'
      router.push('/upload')
    }
  } catch (err) {
    ElMessage.error('加载示例数据失败: ' + err.message)
  }
}

// ==================== 分析执行 ====================
async function handleStartAnalysis() {
  if (!uploadData.value) return

  analyzing.value = true
  currentStep.value = 2
  progress.value = 0
  statusMessage.value = '准备分析...'

  try {
    const variableTypes = { ...editableTypes.value }
    const filteredTypes = {}
    for (const [key, value] of Object.entries(variableTypes)) {
      if (value !== 'exclude') {
        filteredTypes[key] = value
      }
    }

    let sessionId = sessionStore.currentSessionId
    if (!sessionId) {
      sessionId = currentSessionId.value
    }
    if (!sessionId) {
      ElMessage.error('会话ID丢失，请重新上传')
      return
    }

    localStorage.setItem('lastSessionId', sessionId)

    const success = await analysisStore.runAnalysis(sessionId, filteredTypes)

    if (success) {
      ElMessage.success('分析完成！')
    } else {
      ElMessage.error(analysisStore.error || '分析失败')
    }
  } catch (err) {
    ElMessage.error('分析失败: ' + err.message)
  } finally {
    analyzing.value = false
  }
}

async function handleStartDbAnalysis() {
  if (!dbData.value) {
    ElMessage.warning('请先加载数据库表')
    return
  }

  // 如果有多表关系但用户还未确认，提醒
  if (candidateRelations.value.length > 0 && !relationsConfirmed.value) {
    ElMessage.warning('请先确认表间关系')
    return
  }

  analyzing.value = true
  currentStep.value = 2
  progress.value = 0
  statusMessage.value = '准备分析...'

  try {
    // 收集所有表的变量类型
    const allTypes = {}
    for (const [name, table] of Object.entries(dbData.value.tables)) {
      for (const [field, type] of Object.entries(table.variable_types || {})) {
        // 表名.字段名 作为唯一标识
        allTypes[`${name}.${field}`] = type
      }
    }

    let sessionId = sessionStore.currentSessionId
    if (!sessionId) {
      ElMessage.error('会话ID丢失，请重新加载')
      return
    }

    localStorage.setItem('lastSessionId', sessionId)

    const success = await analysisStore.runAnalysis(sessionId, allTypes)

    if (success) {
      ElMessage.success('分析完成！')
    } else {
      ElMessage.error(analysisStore.error || '分析失败')
    }
  } catch (err) {
    ElMessage.error('分析失败: ' + err.message)
  } finally {
    analyzing.value = false
  }
}

function formatProgress(percentage) {
  return `${percentage}%`
}

// ==================== 计算属性 ====================
const totalRows = computed(() => {
  if (!dbData.value) return 0
  return Object.values(dbData.value.tables).reduce((sum, t) => sum + (t.rows || 0), 0)
})

const totalColumns = computed(() => {
  if (!dbData.value) return 0
  return Object.values(dbData.value.tables).reduce((sum, t) => sum + (t.columns || 0), 0)
})
</script>

<style scoped>
.upload-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}
.upload-area {
  background: #fff;
  border-radius: 8px;
  padding: 30px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.drop-zone {
  border: 2px dashed #d9d9d9;
  border-radius: 8px;
  padding: 40px 20px;
  transition: all 0.3s;
}
.drop-zone:hover {
  border-color: #409eff;
}
.file-preview {
  padding: 10px 0;
}
.file-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.file-name {
  font-weight: 500;
  font-size: 16px;
}
.file-stats {
  display: flex;
  gap: 24px;
  color: #666;
  font-size: 14px;
  margin-bottom: 20px;
}
.field-types {
  margin-bottom: 20px;
}
.type-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 8px 16px;
  margin-top: 8px;
}
.type-item {
  display: flex;
  align-items: center;
  gap: 8px;
}
.field-name {
  font-size: 13px;
  font-weight: 500;
  min-width: 80px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.data-preview h4 {
  margin-bottom: 12px;
}
.actions {
  display: flex;
  gap: 12px;
  margin-top: 20px;
}
.progress-section {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #e4e7ed;
}
.status-message {
  margin-top: 8px;
  color: #909399;
  font-size: 14px;
}
.demo-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
}
.demo-card {
  cursor: pointer;
  text-align: center;
  transition: transform 0.2s;
}
.demo-card:hover {
  transform: translateY(-4px);
}
.demo-icon {
  font-size: 40px;
  margin-bottom: 8px;
}
.demo-name {
  font-weight: 600;
  font-size: 16px;
}
.demo-desc {
  font-size: 13px;
  color: #909399;
  margin: 4px 0;
}
.demo-meta {
  font-size: 12px;
  color: #c0c4cc;
}
.table-tags {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
}
.table-switcher {
  margin: 12px 0;
}
.table-info {
  margin: 12px 0;
}
.loading-progress {
  margin: 16px 0;
  padding: 16px;
  background: #f5f7fa;
  border-radius: 8px;
}
</style>