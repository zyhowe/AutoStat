<template>
  <div class="data-traceability">
    <!-- 空状态 -->
    <div v-if="!hasContext && !isLoading && !hasData" class="empty-state">
      <el-empty description="暂无追溯上下文，请从「分析摘要」点击规则/公司/字段名称，或从「数据解码」点击追溯按钮">
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
    <div v-else class="traceability-viewer">
      <!-- ===== 顶部信息栏 ===== -->
      <div class="viewer-header">
        <div class="header-left">
          <span class="context-title">📌 {{ contextDescription || '数据追溯' }}</span>
          <el-tag v-if="sourceType === 'parquet'" size="small" type="primary">📁 样本</el-tag>
          <el-tag v-else-if="sourceType === 'sql'" size="small" type="warning">🗄️ 全量</el-tag>
          <el-tag v-if="rowCount > 0" size="small" type="info">{{ rowCount }} 行</el-tag>
          <el-tag v-if="isComplete && sourceType === 'sql'" size="small" type="success">已加载完成</el-tag>
          <el-tag v-else-if="isLoading" size="small" type="warning">加载中...</el-tag>
          <el-tag v-if="isCancelled" size="small" type="info">已取消</el-tag>
          <!-- 可信度标记 -->
          <el-tag v-if="showTrustLevel && sourceType === 'sql'" size="small" :type="trustTagType">
            {{ trustLabel }}
          </el-tag>
        </div>
        <div class="header-right">
          <!-- 数据源切换 -->
          <el-radio-group v-model="sourceType" size="small" @change="onSourceChange">
            <el-radio-button value="parquet">📁 样本</el-radio-button>
            <el-radio-button value="sql" :disabled="!hasDbConfig">🗄️ 全量</el-radio-button>
          </el-radio-group>
          <el-button size="small" @click="handleRefresh">🔄 刷新</el-button>
          <el-button size="small" type="danger" plain @click="handleStop" :disabled="!isLoading">
            停止
          </el-button>
          <el-button size="small" type="primary" plain @click="handleExport" :disabled="rowCount === 0">
            📥 导出 CSV
          </el-button>
        </div>
      </div>

      <!-- ===== 对比状态栏 ===== -->
      <div class="compare-bar" v-if="showCompare">
        <div class="compare-item parquet">
          <span class="compare-label">📁 样本匹配</span>
          <span class="compare-value">{{ parquetMatchCount }}</span>
          <span class="compare-unit">行</span>
        </div>
        <div class="compare-arrow">↔</div>
        <div class="compare-item sql">
          <span class="compare-label">🗄️ 全量匹配</span>
          <span class="compare-value">{{ sqlMatchCount }}</span>
          <span class="compare-unit">行</span>
        </div>
        <div class="compare-diff" v-if="parquetMatchCount > 0 && sqlMatchCount > 0">
          <span v-if="sqlMatchCount > parquetMatchCount * 10" class="diff-warning">
            ⚠️ 全量是样本的 {{ Math.round(sqlMatchCount / parquetMatchCount) }} 倍
          </span>
          <span v-else-if="sqlMatchCount > parquetMatchCount * 2" class="diff-warn">
            ⚠️ 全量是样本的 {{ Math.round(sqlMatchCount / parquetMatchCount) }} 倍
          </span>
          <span v-else-if="Math.abs(sqlMatchCount - parquetMatchCount) <= 5" class="diff-ok">
            ✅ 样本与全量结果一致
          </span>
          <span v-else class="diff-info">
            全量 {{ sqlMatchCount }} 行 / 样本 {{ parquetMatchCount }} 行
          </span>
        </div>
        <div v-else-if="parquetMatchCount > 0 && sqlMatchCount === 0" class="diff-warning">
          ⚠️ 全量中未找到匹配记录，样本可能包含脏数据
        </div>
        <div v-else-if="parquetMatchCount === 0 && sqlMatchCount > 0" class="diff-info">
          ℹ️ 全量中找到 {{ sqlMatchCount }} 条，样本中无匹配
        </div>
      </div>

      <!-- ===== 状态栏 ===== -->
      <div class="status-bar">
        <span class="status-item">
          <span class="label">数据源</span>
          <span class="value">{{ sourceType === 'parquet' ? '样本 (Parquet)' : '全量 (SQL Server)' }}</span>
        </span>
        <span class="status-item">
          <span class="label">已加载</span>
          <span class="value">{{ rowCount }}</span>
          <span class="unit">行</span>
        </span>
        <span class="status-item" v-if="sourceType === 'sql'">
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
        <span class="status-item" v-if="sourceType === 'parquet' && totalParquetRows > 0">
          <span class="label">总样本</span>
          <span class="value">{{ totalParquetRows }}</span>
          <span class="unit">行</span>
        </span>
      </div>

      <!-- ===== SQL 显示（调试用） ===== -->
      <div v-if="currentSql && sourceType === 'sql'" class="sql-display">
        <el-collapse>
          <el-collapse-item title="📝 查看 SQL" name="sql">
            <pre class="sql-code">{{ currentSql }}</pre>
          </el-collapse-item>
        </el-collapse>
      </div>

      <!-- ===== 错误信息 ===== -->
      <div v-if="errorMessage" class="error-display">
        <el-alert :title="errorMessage" type="error" show-icon :closable="false" />
      </div>

      <!-- ===== 数据表格 ===== -->
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

    <!-- ===== 行详情抽屉 ===== -->
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

        <el-divider />

        <div class="related-diagnostics">
          <div class="section-title">🔗 关联诊断记录</div>
          <div v-if="relatedDiagnostics.length === 0" class="empty-tip">
            该行暂无关联的诊断记录
          </div>
          <el-table v-else :data="relatedDiagnostics" border size="small" max-height="300">
            <el-table-column prop="scenario_id" label="场景" width="80" align="center">
              <template #default="{ row }">
                <el-tag size="small" :type="row.scenario_id === 'E1' ? 'danger' : row.scenario_id === 'E2' ? 'warning' : 'info'">
                  {{ row.scenario_id }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="record_type" label="记录类型" width="100" align="center">
              <template #default="{ row }">
                <el-tag size="small" type="info">{{ row.record_type }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="rule_or_field" label="规则/字段" min-width="150" show-overflow-tooltip />
            <el-table-column prop="severity" label="严重程度" width="90" align="center">
              <template #default="{ row }">
                <el-tag v-if="row.severity" :type="row.severity === 'high' ? 'danger' : row.severity === 'medium' ? 'warning' : 'info'" size="small">
                  {{ row.severity }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="80" align="center">
              <template #default="{ row }">
                <el-button size="small" text type="primary" @click="emit('go-to-diagnostic', row)">查看</el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
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
  // 追溯上下文（来自分析摘要或数据解码）
  context: {
    type: Object,
    default: null
  },
  // 会话 ID
  sessionId: {
    type: String,
    default: ''
  },
  // Parquet 数据（当前上传的数据）
  parquetData: {
    type: Array,
    default: () => []
  },
  // 关联的诊断记录
  diagnostics: {
    type: Array,
    default: () => []
  },
  // 预设描述
  presetDescription: {
    type: String,
    default: ''
  },
  // 是否有数据库配置
  hasDbConfig: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['go-to-decode', 'go-to-diagnostic', 'row-click', 'context-change'])

// ===== 状态 =====
const sourceType = ref('parquet')  // 'parquet' | 'sql'
const tableData = ref([])
const tableColumns = ref([])
const isLoading = ref(false)
const isComplete = ref(false)
const isCancelled = ref(false)
const rowCount = ref(0)
const totalParquetRows = ref(0)
const parquetMatchCount = ref(0)
const sqlMatchCount = ref(0)
const speed = ref(0)
const elapsedTime = ref(0)
const errorMessage = ref('')
const warningMessage = ref('')
const loadingMessage = ref('')
const contextDescription = ref('')
const currentSql = ref('')

// 可信度相关
const trustTagType = ref('info')
const trustLabel = ref('')

const detailDrawerVisible = ref(false)
const selectedRow = ref(null)
const relatedDiagnostics = ref([])

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

const showCompare = computed(() => {
  return parquetMatchCount.value > 0 || sqlMatchCount.value > 0
})

const showTrustLevel = computed(() => {
  return sourceType.value === 'sql' && parquetMatchCount.value > 0 && sqlMatchCount.value > 0
})

const getRowKey = (row) => row.id || row.__row_index

// ===== 表格尺寸 =====
function initTableSize() {
  const el = tableWrapperRef.value
  if (el) {
    const rect = el.getBoundingClientRect()
    tableWidth.value = Math.max(400, rect.width - 4)
    const availableHeight = window.innerHeight - 420
    tableHeight.value = Math.max(200, Math.min(600, availableHeight))
  }
}

// ===== 规则评估器（用于 Parquet 数据） =====
function evaluateRule(rule, fields, row) {
  if (!rule) return false

  // 解析规则: left = right
  let left, right
  if (rule.includes(' = ')) {
    const parts = rule.split(' = ')
    left = parts[0].trim()
    right = parts[1].trim()
  } else if (rule.includes('=')) {
    const parts = rule.split('=')
    left = parts[0].trim()
    right = parts[1].trim()
  } else {
    return false
  }

  // 检查所有字段是否非空
  for (const f of fields) {
    if (row[f] === null || row[f] === undefined) {
      return false
    }
  }

  // 构建可执行的表达式
  // 将字段名替换为 row[field]
  let leftExpr = left
  let rightExpr = right
  const fieldPattern = /companyfixasset\d+/g
  const allFields = [...new Set([...left.match(fieldPattern) || [], ...right.match(fieldPattern) || []])]

  for (const f of allFields) {
    const val = row[f]
    if (val === null || val === undefined) return false
    leftExpr = leftExpr.replace(new RegExp(f, 'g'), `(${val})`)
    rightExpr = rightExpr.replace(new RegExp(f, 'g'), `(${val})`)
  }

  try {
    // 安全评估
    const leftVal = Function('"use strict"; return (' + leftExpr + ')')()
    const rightVal = Function('"use strict"; return (' + rightExpr + ')')()
    const diff = Math.abs(leftVal - rightVal)
    const scale = Math.max(Math.abs(leftVal), Math.abs(rightVal), 1)
    return diff / scale > 0.01
  } catch (e) {
    console.warn('[Parquet过滤] 规则评估失败:', e)
    return false
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

// ===== Parquet 查询 =====
function queryParquet() {
  const context = props.context || {}
  const data = props.parquetData || []

  if (data.length === 0) {
    console.warn('[Parquet查询] 无数据')
    return
  }

  totalParquetRows.value = data.length

  let filtered = []
  let columns = []

  if (context.rule) {
    const rule = context.rule
    const fields = context.fields || []

    // 从规则中提取字段
    const fieldPattern = /companyfixasset\d+/g
    const allFields = [...new Set([...rule.match(fieldPattern) || [], ...fields])]

    if (allFields.length === 0) {
      console.warn('[Parquet查询] 无法从规则中提取字段')
      return
    }

    filtered = data.filter(row => {
      return evaluateRule(rule, allFields, row)
    })

    // 构建列：标识字段 + 规则涉及的字段
    const identityFields = ['id', 'companycode', 'reportdate', 'declaredate']
    const selectFields = [...identityFields]
    for (const f of allFields) {
      if (!selectFields.includes(f) && data[0] && data[0][f] !== undefined) {
        selectFields.push(f)
      }
    }
    columns = selectFields

  } else if (context.row_ids && context.row_ids.length > 0) {
    const ids = context.row_ids
    filtered = data.filter(row => ids.includes(row.id))
    columns = data[0] ? Object.keys(data[0]) : []

  } else if (context.id_range) {
    const start = context.id_range.start || 1
    const end = context.id_range.end || 5000
    filtered = data.filter(row => row.id >= start && row.id <= end)
    columns = data[0] ? Object.keys(data[0]) : []

  } else if (context.company_code) {
    const code = context.company_code
    filtered = data.filter(row => row.companycode === code)
    columns = data[0] ? Object.keys(data[0]) : []

  } else if (context.filters) {
    const filters = context.filters
    filtered = data.filter(row => {
      for (const [key, value] of Object.entries(filters)) {
        if (row[key] !== value) return false
      }
      return true
    })
    columns = data[0] ? Object.keys(data[0]) : []

  } else {
    // 默认：全部数据
    filtered = data
    columns = data[0] ? Object.keys(data[0]) : []
  }

  // 限制行数
  const limit = context.limit || 10000
  if (filtered.length > limit) {
    filtered = filtered.slice(0, limit)
    warningMessage.value = `已达到行数限制 (${limit} 行)，数据被截断`
  }

  parquetMatchCount.value = filtered.length
  tableData.value = filtered
  tableColumns.value = buildColumns(columns)
  rowCount.value = filtered.length
  isComplete.value = true
  isCancelled.value = false
  errorMessage.value = ''
  loadingMessage.value = `加载完成，共 ${filtered.length} 行`

  // 更新可信度
  updateTrustLevel()
}

// ===== SQL 查询 =====
function querySql() {
  if (!props.hasDbConfig) {
    ElMessage.warning('当前会话没有关联的数据库配置')
    return
  }

  tableData.value = []
  tableColumns.value = []
  rowCount.value = 0
  isComplete.value = false
  isCancelled.value = false
  sqlMatchCount.value = 0
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
      console.log('[SQL查询] SQL:', sql)
    },
    onMeta: (columns) => {
      tableColumns.value = buildColumns(columns)
      loadingMessage.value = '已连接，开始接收数据...'
      console.log('[SQL查询] 收到列信息:', columns.length, '列')
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
      sqlMatchCount.value = count || 0
      elapsedTime.value = (Date.now() - startTime.value) / 1000

      if (cancelled) {
        loadingMessage.value = '已取消加载'
        console.log('[SQL查询] 已取消')
      } else {
        loadingMessage.value = `加载完成，共 ${count} 行`
        if (count > 0) {
          ElMessage.success(`已加载 ${count} 行数据`)
        } else {
          ElMessage.info('查询结果为空，没有找到匹配的记录')
        }
        console.log('[SQL查询] 完成，共', count, '行')
        // 更新可信度
        updateTrustLevel()
      }
    },
    onError: (message, isWarning) => {
      console.error('[SQL查询] 错误:', message, isWarning)
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

// ===== 可信度更新 =====
function updateTrustLevel() {
  const parquet = parquetMatchCount.value
  const sql = sqlMatchCount.value

  if (parquet === 0 && sql === 0) {
    trustTagType.value = 'info'
    trustLabel.value = '🟡 均无匹配'
    return
  }

  if (parquet === 0 && sql > 0) {
    trustTagType.value = 'warning'
    trustLabel.value = '🟠 样本未覆盖'
    return
  }

  if (sql === 0 && parquet > 0) {
    trustTagType.value = 'danger'
    trustLabel.value = '🔴 全量未匹配'
    return
  }

  const ratio = sql / parquet
  if (ratio >= 0.9 && ratio <= 1.1) {
    trustTagType.value = 'success'
    trustLabel.value = '🟢 高可信'
  } else if (ratio >= 0.5 && ratio <= 2) {
    trustTagType.value = 'warning'
    trustLabel.value = `🟡 中可信 (${Math.round(ratio * 100)}%)`
  } else if (ratio > 2 && ratio <= 10) {
    trustTagType.value = 'warning'
    trustLabel.value = `🟠 低可信 (全量是样本 ${Math.round(ratio)} 倍)`
  } else if (ratio > 10) {
    trustTagType.value = 'danger'
    trustLabel.value = `🔴 不可信 (全量是样本 ${Math.round(ratio)} 倍)`
  } else {
    trustTagType.value = 'danger'
    trustLabel.value = `🔴 不可信 (样本是全量 ${Math.round(1/ratio)} 倍)`
  }
}

// ===== 数据源切换 =====
function onSourceChange(value) {
  if (value === 'parquet') {
    if (tableData.value.length > 0) {
      // 已有数据，不重新查询
      return
    }
    queryParquet()
  } else if (value === 'sql') {
    if (!props.hasDbConfig) {
      ElMessage.warning('当前会话没有关联的数据库配置')
      sourceType.value = 'parquet'
      return
    }
    querySql()
  }
}

// ===== 操作 =====
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
  if (sourceType.value === 'parquet') {
    queryParquet()
  } else {
    querySql()
  }
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
  a.download = `数据追溯_${sourceType.value}_${new Date().toISOString().slice(0, 10)}.csv`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  ElMessage.success('导出成功')
}

function handleRowClick(row) {
  selectedRow.value = row
  // 查找关联的诊断记录
  if (props.diagnostics && props.diagnostics.length > 0) {
    relatedDiagnostics.value = props.diagnostics.filter(d => d.row === row.id)
  } else {
    relatedDiagnostics.value = []
  }
  detailDrawerVisible.value = true
  emit('row-click', row)
}

// ===== 监听上下文变化 =====
watch(() => props.context, (newContext) => {
  if (newContext && Object.keys(newContext).length > 0) {
    if (isLoading.value) {
      handleStop()
    }
    // 重置状态
    parquetMatchCount.value = 0
    sqlMatchCount.value = 0
    tableData.value = []
    tableColumns.value = []
    rowCount.value = 0
    errorMessage.value = ''
    warningMessage.value = ''

    // 默认先查 Parquet
    sourceType.value = 'parquet'
    setTimeout(() => {
      queryParquet()
      // 如果有数据库配置，自动查询 SQL 做对比（但只查询，不切换）
      if (props.hasDbConfig) {
        setTimeout(() => {
          // 在后台查询 SQL，但不切换源
          const savedSource = sourceType.value
          querySql()
          // 注意：querySql 会修改 sourceType，需要恢复
          // 这里通过一个标志来控制
        }, 500)
      }
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
    setTimeout(() => {
      queryParquet()
    }, 300)
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
  queryParquet,
  querySql,
  handleStop,
  handleRefresh
})
</script>

<style scoped>
.data-traceability {
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

.traceability-viewer {
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
  flex-wrap: wrap;
}

.compare-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 16px;
  background: #f8f9fa;
  border-radius: 8px;
  flex-shrink: 0;
  flex-wrap: wrap;
  margin-bottom: 6px;
  border: 1px solid #e8ecf1;
}
.compare-item {
  display: flex;
  align-items: baseline;
  gap: 4px;
}
.compare-label {
  font-size: 12px;
  color: #909399;
}
.compare-value {
  font-size: 16px;
  font-weight: 700;
  color: #2c3e50;
}
.compare-item.parquet .compare-value {
  color: #409eff;
}
.compare-item.sql .compare-value {
  color: #e6a23c;
}
.compare-unit {
  font-size: 11px;
  color: #909399;
}
.compare-arrow {
  color: #c0c4cc;
  font-size: 18px;
}
.compare-diff {
  font-size: 13px;
  margin-left: 8px;
}
.diff-ok {
  color: #67c23a;
}
.diff-warn {
  color: #e6a23c;
}
.diff-warning {
  color: #f56c6c;
}
.diff-info {
  color: #909399;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 4px 12px;
  background: #f5f7fa;
  border-radius: 6px;
  flex-shrink: 0;
  flex-wrap: wrap;
  margin-bottom: 6px;
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
  margin-bottom: 6px;
}
.sql-code {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 12px 16px;
  border-radius: 6px;
  font-size: 12px;
  overflow-x: auto;
  font-family: 'Consolas', 'Courier New', monospace;
  max-height: 150px;
  overflow-y: auto;
  margin: 0;
}

.error-display {
  margin-bottom: 6px;
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
.section-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 8px;
}
.empty-tip {
  padding: 20px;
  text-align: center;
  color: #909399;
  font-size: 13px;
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
  .compare-bar {
    flex-direction: column;
    align-items: stretch;
    gap: 6px;
  }
  .compare-arrow {
    display: none;
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