<template>
  <div class="stream-sql-viewer">
    <!-- 空状态 -->
    <div v-if="!hasContext && !isLoading && !hasData" class="empty-state">
      <el-empty description="暂无追溯上下文，请从「分析摘要」的排行榜点击规则名称，或从「数据解码」点击追溯按钮来加载数据">
        <el-button type="primary" @click="emit('go-to-decode')">去数据解码</el-button>
      </el-empty>
    </div>

    <!-- 加载中 -->
    <div v-else-if="isLoading && !hasData" class="loading-state">
      <el-skeleton :rows="5" animated />
      <div class="loading-status">
        <el-icon class="is-loading"><Loading /></el-icon>
        <span>{{ loadingMessage || '正在加载数据...' }}</span>
      </div>
    </div>

    <!-- 有数据 -->
    <div v-else class="data-viewer">
      <!-- 顶部信息栏 -->
      <div class="viewer-header">
        <div class="header-left">
          <span class="context-title">📌 {{ contextDescription || '数据追溯' }}</span>
          <el-tag v-if="rowCount > 0" size="small" type="info">
            {{ rowCount }} 行
          </el-tag>
          <el-tag v-if="isComplete" size="small" type="success">已加载完成</el-tag>
          <el-tag v-else-if="isLoading" size="small" type="warning">加载中...</el-tag>
          <el-tag v-if="isCancelled" size="small" type="info">已取消</el-tag>
        </div>
        <div class="header-right">
          <el-button size="small" @click="handleRefresh">🔄 刷新</el-button>
          <el-button size="small" type="danger" plain @click="handleStop" :disabled="!isLoading">
            停止
          </el-button>
          <el-button size="small" type="primary" plain @click="handleExport" :disabled="rowCount === 0">
            📥 导出 CSV
          </el-button>
        </div>
      </div>

      <!-- 状态栏 -->
      <div class="status-bar">
        <span class="status-item">
          <span class="label">已加载</span>
          <span class="value">{{ rowCount }}</span>
          <span class="unit">行</span>
        </span>
        <span class="status-item">
          <span class="label">速度</span>
          <span class="value">{{ speed }}</span>
          <span class="unit">行/秒</span>
        </span>
        <span class="status-item">
          <span class="label">耗时</span>
          <span class="value">{{ elapsedTime }}</span>
          <span class="unit">s</span>
        </span>
        <span class="status-item" v-if="errorMessage">
          <el-tag type="danger" size="small">{{ errorMessage }}</el-tag>
        </span>
        <span class="status-item" v-if="warningMessage">
          <el-tag type="warning" size="small">{{ warningMessage }}</el-tag>
        </span>
      </div>

      <!-- SQL 显示 -->
      <div v-if="currentSql" class="sql-display">
        <el-collapse>
          <el-collapse-item title="📝 查看 SQL" name="sql">
            <pre class="sql-code">{{ currentSql }}</pre>
          </el-collapse-item>
        </el-collapse>
      </div>

      <!-- 错误信息展示 -->
      <div v-if="errorMessage" class="error-display">
        <el-alert
          :title="errorMessage"
          type="error"
          show-icon
          :closable="false"
        />
      </div>

      <!-- 数据表格 -->
      <div class="table-wrapper" ref="tableWrapperRef">
        <el-table-v2
          v-if="tableColumns.length > 0 && tableData.length > 0"
          :data="tableData"
          :columns="tableColumns"
          :row-height="36"
          :header-height="40"
          :width="tableWidth"
          :height="tableHeight"
          fixed
          :row-key="getRowKey"
          @row-click="handleRowClick"
        />
        <div v-else-if="!isLoading && !errorMessage && !hasData" class="empty-table">
          <el-empty description="查询结果为空，没有找到匹配的记录" :image-size="60" />
        </div>
        <div v-else-if="isLoading" class="empty-table">
          <el-empty description="等待数据..." :image-size="40" />
        </div>
      </div>
    </div>

    <!-- 行详情抽屉 -->
    <el-drawer
      v-model="detailDrawerVisible"
      :title="`📋 行 ${selectedRow?.id || ''} 详情`"
      size="50%"
      destroy-on-close
    >
      <div v-if="selectedRow" class="detail-panel">
        <el-descriptions :column="1" border size="small">
          <el-descriptions-item
            v-for="(value, key) in selectedRow"
            :key="key"
            :label="key"
          >
            <span v-if="value === null || value === undefined" style="color: #ccc;">(空)</span>
            <span v-else-if="typeof value === 'number' && Number.isFinite(value)">
              {{ Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4) }}
            </span>
            <span v-else-if="typeof value === 'string' && value.length > 200">
              {{ value.slice(0, 200) }}...
            </span>
            <span v-else>{{ String(value) }}</span>
          </el-descriptions-item>
        </el-descriptions>
      </div>
    </el-drawer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick, h } from 'vue'
import { ElMessage } from 'element-plus'
import { Loading } from '@element-plus/icons-vue'
import { streamApi } from '../api'

const props = defineProps({
  context: { type: Object, default: null },
  sessionId: { type: String, default: '' },
  presetDescription: { type: String, default: '' }
})

const emit = defineEmits(['go-to-decode', 'row-click', 'context-change'])

// ===== 状态 =====
const tableData = ref([])
const tableColumns = ref([])
const isLoading = ref(false)
const isComplete = ref(false)
const isCancelled = ref(false)
const rowCount = ref(0)
const totalEstimate = ref(null)
const speed = ref(0)
const elapsedTime = ref(0)
const errorMessage = ref('')
const warningMessage = ref('')
const loadingMessage = ref('')
const contextDescription = ref('')
const currentSql = ref('')

const detailDrawerVisible = ref(false)
const selectedRow = ref(null)

const startTime = ref(null)
const lastRowTime = ref(null)
const speedSamples = ref([])

let abortController = null
let resizeObserver = null

const tableWrapperRef = ref(null)
const tableWidth = ref(800)
const tableHeight = ref(400)

// ===== 计算属性 =====
const hasContext = computed(() => {
  return props.context && Object.keys(props.context).length > 0
})

const hasData = computed(() => tableData.value.length > 0)

const getRowKey = (row) => row.id || row.__row_index

// ===== 表格尺寸 =====
function initTableSize() {
  const el = tableWrapperRef.value
  if (el) {
    const rect = el.getBoundingClientRect()
    tableWidth.value = Math.max(400, rect.width - 4)
    const availableHeight = window.innerHeight - 380
    tableHeight.value = Math.max(200, Math.min(600, availableHeight))
  }
}

// ===== 构建表格列 =====
function buildColumns(columns) {
  if (!columns || columns.length === 0) return []

  const displayCols = columns.slice(0, 30)

  return displayCols.map(col => ({
    key: col,
    title: col,
    dataKey: col,
    width: Math.max(120, Math.min(280, col.length * 12 + 40)),
    cellRenderer: ({ cellData }) => {
      if (cellData === null || cellData === undefined) {
        return h('span', { style: { color: '#ccc' } }, '(空)')
      }
      if (typeof cellData === 'number' && Number.isFinite(cellData)) {
        if (Number.isInteger(cellData)) {
          return h('span', {}, cellData.toLocaleString())
        }
        return h('span', {}, cellData.toFixed(4))
      }
      if (typeof cellData === 'string' && cellData.length > 100) {
        return h('span', { title: cellData }, cellData.slice(0, 100) + '...')
      }
      return h('span', {}, String(cellData))
    }
  }))
}

// ===== 查询方法 =====
function startQuery() {
  tableData.value = []
  tableColumns.value = []
  rowCount.value = 0
  isComplete.value = false
  isCancelled.value = false
  errorMessage.value = ''
  warningMessage.value = ''
  speed.value = 0
  elapsedTime.value = 0
  speedSamples.value = []
  startTime.value = Date.now()
  lastRowTime.value = Date.now()

  abortController = new AbortController()
  isLoading.value = true
  loadingMessage.value = '正在连接数据库...'

  console.log('[全量数据解码] 开始查询，context:', props.context)

  const context = props.context || {}

  streamApi.streamQuery({
    sessionId: props.sessionId,
    context: context,
    batchSize: 100,
    maxRows: 10000,
    abortController: abortController,
    onInfo: (description, sql) => {
      contextDescription.value = description || props.presetDescription || '数据追溯'
      currentSql.value = sql
      loadingMessage.value = '正在执行查询...'
      console.log('[全量数据解码] SQL:', sql)
    },
    onMeta: (columns) => {
      tableColumns.value = buildColumns(columns)
      loadingMessage.value = `已连接，开始接收数据...`
      console.log('[全量数据解码] 收到列信息:', columns.length, '列')
      nextTick(() => { initTableSize() })
    },
    onChunk: (row, count) => {
      tableData.value.push(row)
      rowCount.value = count

      const now = Date.now()
      const delta = (now - lastRowTime.value) / 1000
      if (delta > 0) {
        const currentSpeed = 1 / delta
        speedSamples.value.push(currentSpeed)
        if (speedSamples.value.length > 10) {
          speedSamples.value.shift()
        }
        speed.value = Math.round(
          speedSamples.value.reduce((a, b) => a + b, 0) / speedSamples.value.length
        )
      }
      lastRowTime.value = now
      elapsedTime.value = (now - startTime.value) / 1000

      nextTick(() => {
        const tableEl = document.querySelector('.el-table-v2')
        if (tableEl) {
          const bodyEl = tableEl.querySelector('.el-table-v2__body')
          if (bodyEl) {
            bodyEl.scrollTop = bodyEl.scrollHeight
          }
        }
      })
    },
    onComplete: (count, cancelled) => {
      isLoading.value = false
      isComplete.value = true
      isCancelled.value = cancelled || false
      elapsedTime.value = (Date.now() - startTime.value) / 1000

      if (cancelled) {
        loadingMessage.value = '已取消加载'
        console.log('[全量数据解码] 已取消')
      } else {
        loadingMessage.value = `加载完成，共 ${count} 行`
        if (count > 0) {
          ElMessage.success(`已加载 ${count} 行数据`)
        } else {
          ElMessage.info('查询结果为空，没有找到匹配的记录')
        }
        console.log('[全量数据解码] 完成，共', count, '行')
      }
    },
    onError: (message, isWarning) => {
      console.error('[全量数据解码] 错误:', message, isWarning)
      if (isWarning) {
        warningMessage.value = message
        ElMessage.warning(message)
      } else {
        errorMessage.value = message
        isLoading.value = false
        ElMessage.error(message)
      }
    }
  })
}

function handleStop() {
  if (abortController) {
    abortController.abort()
    isLoading.value = false
    isCancelled.value = true
    loadingMessage.value = '已停止加载'
    ElMessage.info('已停止加载')
  }
}

function handleRefresh() {
  if (isLoading.value) {
    ElMessage.warning('正在加载中，请先停止')
    return
  }
  startQuery()
}

function handleExport() {
  if (tableData.value.length === 0) {
    ElMessage.warning('没有数据可导出')
    return
  }

  const columns = tableColumns.value.map(c => c.key)
  const header = columns.join(',')
  const body = tableData.value.map(row => {
    return columns.map(col => {
      const val = row[col]
      if (val === null || val === undefined) return ''
      if (typeof val === 'string' && (val.includes(',') || val.includes('"'))) {
        return `"${val.replace(/"/g, '""')}"`
      }
      if (typeof val === 'number') {
        return String(val)
      }
      return String(val)
    }).join(',')
  }).join('\n')

  const csv = header + '\n' + body
  const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `全量数据_${new Date().toISOString().slice(0, 10)}.csv`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  ElMessage.success('导出成功')
}

function handleRowClick(row) {
  selectedRow.value = row
  detailDrawerVisible.value = true
  emit('row-click', row)
}

// ===== 监听上下文变化 =====
watch(() => props.context, (newContext) => {
  if (newContext && Object.keys(newContext).length > 0) {
    if (isLoading.value) {
      handleStop()
    }
    setTimeout(() => {
      startQuery()
    }, 300)
  }
}, { deep: true })

// ===== 生命周期 =====
onMounted(() => {
  const el = tableWrapperRef.value
  if (el) {
    resizeObserver = new ResizeObserver(() => {
      initTableSize()
    })
    resizeObserver.observe(el)
  }
  window.addEventListener('resize', initTableSize)
  setTimeout(initTableSize, 200)

  if (props.context && Object.keys(props.context).length > 0) {
    startQuery()
  }
})

onUnmounted(() => {
  if (abortController) {
    abortController.abort()
  }
  if (resizeObserver) {
    resizeObserver.disconnect()
  }
  window.removeEventListener('resize', initTableSize)
})

defineExpose({
  startQuery,
  handleStop,
  handleRefresh
})
</script>

<style scoped>
.stream-sql-viewer {
  height: 100%;
  display: flex;
  flex-direction: column;
  min-height: 300px;
}

.empty-state {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 60px 0;
}

.loading-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 0;
}
.loading-status {
  margin-top: 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  color: #909399;
  font-size: 14px;
}
.loading-status .el-icon {
  font-size: 20px;
  color: #409eff;
}
.is-loading {
  animation: rotate 1s linear infinite;
}
@keyframes rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.data-viewer {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  flex-shrink: 0;
  flex-wrap: wrap;
  gap: 8px;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.context-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
}
.header-right {
  display: flex;
  gap: 6px;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 6px 12px;
  background: #f5f7fa;
  border-radius: 6px;
  flex-shrink: 0;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.status-item {
  display: flex;
  align-items: baseline;
  gap: 2px;
  font-size: 12px;
}
.status-item .label {
  color: #909399;
}
.status-item .value {
  font-weight: 600;
  color: #2c3e50;
}
.status-item .unit {
  color: #909399;
  font-size: 11px;
}

.sql-display {
  margin-bottom: 8px;
}
.sql-code {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 12px 16px;
  border-radius: 6px;
  font-size: 12px;
  overflow-x: auto;
  font-family: 'Consolas', 'Courier New', monospace;
  max-height: 200px;
  overflow-y: auto;
  margin: 0;
}

.error-display {
  margin-bottom: 8px;
}

.table-wrapper {
  flex: 1;
  overflow: hidden;
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  min-height: 200px;
  width: 100%;
}
.empty-table {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  min-height: 200px;
}

.detail-panel {
  padding: 4px 0;
}

@media (max-width: 768px) {
  .viewer-header {
    flex-direction: column;
    align-items: flex-start;
  }
  .header-right {
    width: 100%;
    justify-content: flex-start;
    flex-wrap: wrap;
  }
  .status-bar {
    gap: 10px;
    padding: 8px 10px;
  }
  .table-wrapper {
    min-height: 150px;
  }
}
</style>