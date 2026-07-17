<template>
  <div class="upload-page">
    <!-- ===== 步骤条 ===== -->
    <el-steps :active="stepIndex" finish-status="success" align-center style="margin-bottom: 30px;">
      <el-step title="选择数据源" />
      <el-step title="数据预览" />
      <el-step title="开始分析" />
    </el-steps>

    <!-- ============================================================ -->
    <!-- Step 1: 选择数据源 -->
    <!-- ============================================================ -->
    <div v-if="currentStep === 'source'" class="source-step">
      <el-radio-group v-model="dataSourceType" size="large" style="margin-bottom: 20px;" @change="onDataSourceChange">
        <el-radio-button value="file">📁 文件上传</el-radio-button>
        <el-radio-button value="database">🗄️ 数据库</el-radio-button>
        <el-radio-button value="demo">📊 示例数据</el-radio-button>
      </el-radio-group>

      <!-- ===== 文件上传 ===== -->
      <div v-if="dataSourceType === 'file'" class="upload-area">
        <!-- 上传拖拽区 -->
        <div class="drop-zone" @dragover.prevent @drop.prevent="handleDrop">
          <el-upload
            ref="uploadRef"
            drag
            :auto-upload="false"
            :on-change="handleFileSelect"
            :on-remove="handleFileRemove"
            :limit="50"
            multiple
            action="#"
          >
            <el-icon class="el-icon--upload"><Upload /></el-icon>
            <div class="el-upload__text">
              拖拽文件到此处，或 <em>点击选择</em>（支持多选，可分批添加）
            </div>
            <template #tip>
              <div class="el-upload__tip">
                支持 CSV, Excel, JSON, TXT, Parquet 格式，单文件不超过 100MB，最多 50 个文件
              </div>
            </template>
          </el-upload>
        </div>

        <!-- ===== 已选文件列表 ===== -->
        <div v-if="selectedFiles.length > 0" class="selected-files">
          <div class="files-header">
            <span class="files-title">📋 已选文件（{{ selectedFiles.length }} 个）</span>
            <div class="files-actions">
              <el-button size="small" type="danger" plain @click="clearAllFiles">清空所有</el-button>
              <el-button
                size="small"
                type="primary"
                :loading="uploading"
                :disabled="uploading"
                @click="handleConfirmUpload"
              >
                {{ uploading ? '上传中...' : '✅ 确认上传' }}
              </el-button>
            </div>
          </div>

          <el-table :data="selectedFiles" border size="small" max-height="200">
            <el-table-column prop="name" label="文件名" min-width="200" show-overflow-tooltip />
            <el-table-column prop="size" label="大小" width="120" align="center">
              <template #default="{ row }">
                {{ formatFileSize(row.size) }}
              </template>
            </el-table-column>
            <el-table-column label="状态" width="120" align="center">
              <template #default="{ row }">
                <el-tag v-if="row.status === 'pending'" size="small" type="info">待上传</el-tag>
                <el-tag v-else-if="row.status === 'uploading'" size="small" type="warning">上传中</el-tag>
                <el-tag v-else-if="row.status === 'success'" size="small" type="success">✅ 已上传</el-tag>
                <el-tag v-else-if="row.status === 'error'" size="small" type="danger">❌ 失败</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="80" align="center">
              <template #default="{ row }">
                <el-button size="small" type="danger" text @click="removeFile(row)">移除</el-button>
              </template>
            </el-table-column>
          </el-table>

          <div class="files-footer">
            <span class="files-tip">💡 可继续添加文件，确认上传后统一处理</span>
          </div>
        </div>

        <!-- 空状态提示 -->
        <div v-else class="empty-files">
          <el-empty description="暂无文件，请选择或拖拽文件到上方区域" :image-size="60" />
        </div>
      </div>

      <!-- ===== 数据库连接 ===== -->
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

        <div v-if="loadingDb" class="loading-progress">
          <el-progress :percentage="dbLoadProgress" />
          <p class="status-message">{{ dbLoadStatus }}</p>
        </div>
      </div>

      <!-- ===== 示例数据 ===== -->
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
            <div class="demo-meta">{{ demo.rows }}</div>
            <el-button type="primary" size="small" style="margin-top: 12px;">加载此数据</el-button>
          </el-card>
        </div>
      </div>
    </div>

    <!-- ============================================================ -->
    <!-- Step 2: 数据预览 -->
    <!-- ============================================================ -->
    <div v-else-if="currentStep === 'preview' && loadedData" class="preview-step">
      <div class="preview-header">
        <div class="header-left">
          <span class="source-name">📊 {{ loadedData.sourceName || '数据' }}</span>
          <el-tag size="small" :type="loadedData.isMultiTable ? 'warning' : 'success'">
            {{ loadedData.isMultiTable ? `${loadedData.tableNames.length} 张表` : '单表' }}
          </el-tag>
        </div>
        <div class="header-right">
          <el-button size="small" @click="goBack">← 返回选择</el-button>
        </div>
      </div>

      <div v-if="loadedData.isMultiTable" class="table-switcher">
        <el-radio-group v-model="selectedTable" size="small" @change="onTableChange">
          <el-radio-button
            v-for="name in loadedData.tableNames"
            :key="name"
            :label="name"
          />
        </el-radio-group>
      </div>

      <div class="table-info" v-if="currentTableInfo">
        <el-descriptions :column="3" border size="small">
          <el-descriptions-item label="表名">{{ selectedTable || '数据表' }}</el-descriptions-item>
          <el-descriptions-item label="行数">{{ currentTableInfo.rows }}</el-descriptions-item>
          <el-descriptions-item label="列数">{{ currentTableInfo.columns }}</el-descriptions-item>
        </el-descriptions>
      </div>

      <div class="field-types">
        <h4>📋 字段类型调整 <el-tag size="small" type="info">修改后自动保存</el-tag></h4>
        <div class="type-grid">
          <div v-for="(type, field) in currentFieldTypes" :key="field" class="type-item">
            <span class="field-name">{{ field }}</span>
            <el-select
              v-model="currentFieldTypes[field]"
              size="small"
              style="width: 120px;"
              @change="onFieldTypeChange(field, $event)"
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

      <TableRelationConfirm
        v-if="loadedData.isMultiTable"
        :relations="loadedData.candidateRelations || []"
        :table-names="loadedData.tableNames"
        :table-columns="loadedData.tableColumns || {}"
        :confirmed="relationsConfirmed"
        @confirm="handleRelationsConfirmed"
        @skip="handleRelationsSkipped"
        @edit="handleRelationsEdit"
      />

      <div class="actions">
        <el-button
          type="primary"
          size="large"
          :loading="analysisRunning"
          :disabled="!canStartAnalysis"
          @click="handleStartAnalysis"
        >
          {{ analysisRunning ? '分析中...' : '🚀 开始分析' }}
        </el-button>
        <el-tag v-if="loadedData.isMultiTable && !relationsConfirmed" type="warning" size="small">
          ⚠️ 请先确认表间关系
        </el-tag>
        <el-tag v-else-if="loadedData.isMultiTable && relationsConfirmed" type="success" size="small">
          ✅ 关系已确认
        </el-tag>
      </div>
    </div>

    <!-- ============================================================ -->
    <!-- Step 3: 分析进度 -->
    <!-- ============================================================ -->
    <div v-else-if="currentStep === 'analyzing'" class="analyzing-step">
      <div class="progress-card">
        <h3>⏳ 分析进行中</h3>
        <el-progress :percentage="progress" :format="formatProgress" />
        <p class="status-message">{{ statusMessage }}</p>
        <el-button v-if="progress === 100" type="primary" @click="goToReport">查看报告 →</el-button>
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

// ============================================================
// 状态
// ============================================================

const currentStep = ref('source')
const stepIndex = computed(() => {
  const map = { source: 0, preview: 1, analyzing: 2 }
  return map[currentStep.value] || 0
})

const dataSourceType = ref('file')
const loadedData = ref(null)
const selectedTable = ref('')
const currentFieldTypes = ref({})
const currentPreviewData = ref([])
const currentPreviewColumns = ref([])
const relationsConfirmed = ref(false)
const analysisRunning = ref(false)
const progress = ref(0)
const statusMessage = ref('')
const uploadRef = ref(null)

// ===== 文件上传状态 =====
const selectedFiles = ref([])
const uploading = ref(false)

// 数据库配置
const dbConfigs = ref([])
const dbForm = reactive({
  configName: '',
  tableNamesInput: '',
  limit: 5000
})
const parsedTableNames = ref([])
const loadingDb = ref(false)
const dbLoadProgress = ref(0)
const dbLoadStatus = ref('')

const demoDatasets = [
  { key: 'sales', name: '销售数据', icon: '📊', description: '零售销售分析', rows: '5,000 行 × 9 列' },
  { key: 'user', name: '用户数据', icon: '👥', description: '用户行为分析', rows: '3,000 行 × 10 列' },
  { key: 'medical', name: '医疗数据', icon: '🏥', description: '患者就诊分析', rows: '2,000 行 × 11 列' },
  { key: 'ecommerce', name: '电商多表', icon: '🏪', description: '订单+客户+明细 3表关联', rows: '3 张表' }
]

// ============================================================
// 计算属性
// ============================================================

const canStartAnalysis = computed(() => {
  if (!loadedData.value) return false
  if (loadedData.value.isMultiTable && !relationsConfirmed.value) return false
  return true
})

const currentTableInfo = computed(() => {
  if (!loadedData.value || !selectedTable.value) return null
  const table = loadedData.value.tables[selectedTable.value]
  if (!table) return null
  return { rows: table.rows, columns: table.columns }
})

// ============================================================
// 生命周期
// ============================================================

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

// ============================================================
// 统一数据加载
// ============================================================

function normalizeLoadedData(response, sourceName = '数据') {
  const result = {
    sessionId: response.session_id,
    sourceName: sourceName || response.source_name || '数据',
    isMultiTable: false,
    tableNames: [],
    tables: {},
    tableColumns: {},
    candidateRelations: response.candidate_relations || [],
    singleTablePreview: null
  }

  if (response.tables && Object.keys(response.tables).length > 1) {
    result.isMultiTable = true
    result.tableNames = Object.keys(response.tables)
    for (const [name, info] of Object.entries(response.tables)) {
      const preview = info.preview || { head: [], columns: [] }
      result.tables[name] = {
        rows: info.rows,
        columns: info.columns,
        variable_types: info.variable_types || {},
        preview: preview,
        field_types_cache: {}
      }
      result.tableColumns[name] = preview.columns || []
    }
  } else if (response.rows !== undefined) {
    result.isMultiTable = false
    result.tableNames = ['数据表']
    const preview = response.preview || { head: [], columns: [] }
    result.tables['数据表'] = {
      rows: response.rows,
      columns: response.columns,
      variable_types: response.variable_types || {},
      preview: preview,
      field_types_cache: {}
    }
    result.tableColumns['数据表'] = preview.columns || []
    result.singleTablePreview = preview
  } else if (response.tables && Object.keys(response.tables).length === 1) {
    const [name, info] = Object.entries(response.tables)[0]
    result.isMultiTable = false
    result.tableNames = [name]
    const preview = info.preview || { head: [], columns: [] }
    result.tables[name] = {
      rows: info.rows,
      columns: info.columns,
      variable_types: info.variable_types || {},
      preview: preview,
      field_types_cache: {}
    }
    result.tableColumns[name] = preview.columns || []
  }

  result.candidateRelations = result.candidateRelations || []
  return result
}

function setLoadedData(response, sourceName) {
  const normalized = normalizeLoadedData(response, sourceName)

  if (normalized.sessionId) {
    sessionStore.currentSessionId = normalized.sessionId
    localStorage.setItem('lastSessionId', normalized.sessionId)
    sessionStore.loadProjects().catch(err => console.warn('刷新项目列表失败:', err))
  }

  loadedData.value = normalized
  selectedTable.value = normalized.tableNames[0] || ''
  relationsConfirmed.value = false

  updateCurrentTable()
  currentStep.value = 'preview'
}

function updateCurrentTable() {
  if (!loadedData.value || !selectedTable.value) return

  const table = loadedData.value.tables[selectedTable.value]
  if (!table) return

  const cachedTypes = table.field_types_cache || {}
  const initialTypes = table.variable_types || {}
  currentFieldTypes.value = { ...initialTypes, ...cachedTypes }

  currentPreviewData.value = table.preview.head || []
  currentPreviewColumns.value = table.preview.columns || []
}

// ============================================================
// 字段类型更新
// ============================================================

async function onFieldTypeChange(field, newType) {
  if (!loadedData.value || !selectedTable.value) return

  currentFieldTypes.value[field] = newType

  try {
    await dataApi.updateFieldTypes(
      loadedData.value.sessionId,
      selectedTable.value,
      currentFieldTypes.value
    )
    if (loadedData.value.tables[selectedTable.value]) {
      loadedData.value.tables[selectedTable.value].field_types_cache = { ...currentFieldTypes.value }
    }
  } catch (err) {
    ElMessage.error('保存字段类型失败: ' + err.message)
    const table = loadedData.value.tables[selectedTable.value]
    if (table) {
      currentFieldTypes.value = { ...table.variable_types }
    }
  }
}

// ============================================================
// 表切换
// ============================================================

function onTableChange(value) {
  selectedTable.value = value
  updateCurrentTable()
}

// ============================================================
// 关系确认
// ============================================================

async function handleRelationsConfirmed(relations) {
  if (!loadedData.value) return

  try {
    await dataApi.confirmRelations({
      session_id: loadedData.value.sessionId,
      relationships: relations
    })
    relationsConfirmed.value = true
    loadedData.value.candidateRelations = relations
    ElMessage.success(relations.length > 0
      ? `已确认 ${relations.length} 条表间关系`
      : '已确认无表间关系，将作为独立表分析'
    )
  } catch (err) {
    ElMessage.error('保存关系失败: ' + err.message)
  }
}

function handleRelationsSkipped() {
  relationsConfirmed.value = true
  ElMessage.info('已跳过关系配置')
}

function handleRelationsEdit() {
  relationsConfirmed.value = false
}

// ============================================================
// 文件选择（只累加文件，不上传）
// ============================================================

function handleFileSelect(file, fileList) {
  // file 是当前添加的文件，fileList 是所有文件
  // 但我们用 selectedFiles 独立维护列表
  // 检查是否已存在
  const exists = selectedFiles.value.some(f => f.name === file.name && f.size === file.size)
  if (!exists) {
    selectedFiles.value.push({
      name: file.name,
      size: file.size,
      raw: file.raw,
      status: 'pending'
    })
  }
}

function handleFileRemove(file, fileList) {
  // 从 selectedFiles 中移除
  const index = selectedFiles.value.findIndex(f => f.name === file.name && f.size === file.size)
  if (index > -1) {
    selectedFiles.value.splice(index, 1)
  }
}

function removeFile(row) {
  const index = selectedFiles.value.indexOf(row)
  if (index > -1) {
    selectedFiles.value.splice(index, 1)
  }
  // 同时从 upload 组件中移除
  if (uploadRef.value) {
    const uploadFiles = uploadRef.value.uploadFiles
    const f = uploadFiles.find(u => u.name === row.name && u.size === row.size)
    if (f) {
      uploadRef.value.handleRemove(f)
    }
  }
}

function clearAllFiles() {
  selectedFiles.value = []
  if (uploadRef.value) {
    uploadRef.value.clearFiles()
  }
}

function handleDrop(e) {
  const files = e.dataTransfer.files
  if (files.length === 0) return
  e.preventDefault()

  // 拖拽的文件直接添加到 selectedFiles
  for (const file of files) {
    const exists = selectedFiles.value.some(f => f.name === file.name && f.size === file.size)
    if (!exists) {
      selectedFiles.value.push({
        name: file.name,
        size: file.size,
        raw: file,
        status: 'pending'
      })
    }
  }
}

// ============================================================
// 确认上传（创建项目，上传文件）
// ============================================================

async function handleConfirmUpload() {
  if (selectedFiles.value.length === 0) {
    ElMessage.warning('请先选择文件')
    return
  }

  // 检查是否有文件正在上传中
  if (uploading.value) return

  // 检查是否有失败的文件
  const hasError = selectedFiles.value.some(f => f.status === 'error')
  if (hasError) {
    // 允许重试，重置所有失败状态
    selectedFiles.value.forEach(f => {
      if (f.status === 'error') f.status = 'pending'
    })
  }

  uploading.value = true

  try {
    // 获取所有待上传的文件
    const filesToUpload = selectedFiles.value.filter(f => f.status === 'pending')
    const rawFiles = filesToUpload.map(f => f.raw)

    if (rawFiles.length === 0) {
      ElMessage.warning('没有待上传的文件')
      uploading.value = false
      return
    }

    // 标记为上传中
    filesToUpload.forEach(f => f.status = 'uploading')

    // 创建会话（只有第一次创建）
    if (!sessionStore.currentSessionId) {
      await sessionStore.createSession('multi_upload')
    }

    // 执行上传
    let result
    if (rawFiles.length === 1) {
      result = await sessionStore.uploadFile(rawFiles[0])
    } else {
      result = await dataApi.uploadMulti(rawFiles, sessionStore.currentSessionId)
    }

    // 标记为成功
    filesToUpload.forEach(f => f.status = 'success')
    ElMessage.success(`成功上传 ${rawFiles.length} 个文件`)

    if (result) {
      const isMulti = result.tables && Object.keys(result.tables).length > 1
      const sourceName = isMulti
        ? `多文件 (${Object.keys(result.tables).length} 个)`
        : rawFiles[0]?.name || '数据'
      setLoadedData(result, sourceName)
    }

    // 保留已上传的文件在列表中，但标记为成功
    // 用户可以通过清空按钮清空

  } catch (err) {
    // 标记为失败
    selectedFiles.value.forEach(f => {
      if (f.status === 'uploading') f.status = 'error'
    })
    ElMessage.error('上传失败: ' + err.message)
  } finally {
    uploading.value = false
  }
}

function formatFileSize(bytes) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  while (bytes >= 1024 && i < units.length - 1) {
    bytes /= 1024
    i++
  }
  return bytes.toFixed(1) + ' ' + units[i]
}

// ============================================================
// 数据库加载
// ============================================================

function onDbConfigChange() {}

function parseTableNames() {
  const input = dbForm.tableNamesInput || ''
  const parts = input.split(/[,，\s\n]+/).map(s => s.trim())
  parsedTableNames.value = parts.filter(s => s.length > 0)
}

async function handleLoadDbTables() {
  if (!dbForm.configName || parsedTableNames.value.length === 0) {
    ElMessage.warning('请选择配置并输入至少一个表名')
    return
  }

  loadingDb.value = true
  dbLoadProgress.value = 0
  dbLoadStatus.value = '正在加载表...'

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
      setLoadedData(result, `${parsedTableNames.value[0]}_db`)
    }
  } catch (err) {
    ElMessage.error('加载失败: ' + err.message)
  } finally {
    loadingDb.value = false
  }
}

// ============================================================
// 示例数据
// ============================================================

async function handleLoadDemo(datasetKey) {
  try {
    const result = await dataApi.loadDemo(datasetKey)
    if (result && result.session_id) {
      const isMulti = result.tables && Object.keys(result.tables).length > 1
      setLoadedData(result, isMulti ? '电商多表' : `${datasetKey}_demo`)
    }
  } catch (err) {
    ElMessage.error('加载示例数据失败: ' + err.message)
  }
}

// ============================================================
// 开始分析
// ============================================================

async function handleStartAnalysis() {
  if (!loadedData.value) return
  if (loadedData.value.isMultiTable && !relationsConfirmed.value) {
    ElMessage.warning('请先确认表间关系')
    return
  }

  analysisRunning.value = true
  currentStep.value = 'analyzing'
  progress.value = 0
  statusMessage.value = '准备分析...'

  try {
    const success = await analysisStore.runAnalysis(loadedData.value.sessionId)

    if (success) {
      progress.value = 100
      statusMessage.value = '分析完成！'
      ElMessage.success('分析完成！')
      setTimeout(() => {
        router.push('/report-summary')
      }, 1000)
    } else {
      statusMessage.value = '分析失败: ' + (analysisStore.error || '未知错误')
      ElMessage.error(analysisStore.error || '分析失败')
      analysisRunning.value = false
    }
  } catch (err) {
    statusMessage.value = '分析失败: ' + err.message
    ElMessage.error('分析失败: ' + err.message)
    analysisRunning.value = false
  }
}

// ============================================================
// 辅助函数
// ============================================================

function formatProgress(percentage) {
  return `${percentage}%`
}

function goBack() {
  currentStep.value = 'source'
  relationsConfirmed.value = false
  // 清空文件列表
  selectedFiles.value = []
  if (uploadRef.value) {
    uploadRef.value.clearFiles()
  }
}

function goToReport() {
  router.push('/report-summary')
}

function onDataSourceChange() {
  // 切换数据源时重置状态
  selectedFiles.value = []
  if (uploadRef.value) {
    uploadRef.value.clearFiles()
  }
}
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

/* ===== 已选文件列表 ===== */
.selected-files {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid #e4e7ed;
}

.files-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  flex-wrap: wrap;
  gap: 8px;
}

.files-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
}

.files-actions {
  display: flex;
  gap: 8px;
}

.files-footer {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
}

.files-tip {
  font-size: 12px;
  color: #909399;
}

.empty-files {
  padding: 20px 0;
}

.preview-step {
  background: #fff;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #e4e7ed;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.source-name {
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
}

.table-switcher {
  margin: 12px 0;
}
.table-info {
  margin: 12px 0;
}

.field-types {
  margin-bottom: 20px;
}
.type-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 8px 16px;
  margin-top: 8px;
  max-height: 300px;
  overflow-y: auto;
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
  align-items: center;
  gap: 16px;
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #e4e7ed;
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

.loading-progress {
  margin: 16px 0;
  padding: 16px;
  background: #f5f7fa;
  border-radius: 8px;
}

.analyzing-step {
  max-width: 600px;
  margin: 0 auto;
  padding: 40px 0;
}
.progress-card {
  background: #fff;
  border-radius: 12px;
  padding: 40px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  text-align: center;
}
.progress-card h3 {
  margin-bottom: 24px;
  color: #2c3e50;
}
.status-message {
  margin-top: 16px;
  color: #909399;
  font-size: 14px;
}
</style>