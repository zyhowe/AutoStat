<template>
  <div class="data-decode-trace">
    <!-- ===== 顶部控制区 ===== -->
    <div class="control-bar">
      <div class="control-left">
        <!-- 场景筛选 -->
        <el-select
          :model-value="filter.scenario"
          placeholder="按场景筛选"
          clearable
          size="small"
          style="width:160px;"
          @update:model-value="updateFilter('scenario', $event)"
        >
          <el-option label="全部场景" value="" />
          <el-option
            v-for="s in scenarioResults"
            :key="s.scenario_id"
            :label="s.business_name || s.name"
            :value="String(s.scenario_id)"
          />
        </el-select>

        <!-- 记录类型 -->
        <el-select
          :model-value="filter.recordType"
          placeholder="记录类型"
          clearable
          size="small"
          style="width:110px;"
          @update:model-value="updateFilter('recordType', $event)"
        >
          <el-option label="全部" value="" />
          <el-option label="勾稽" value="violation" />
          <el-option label="异常" value="outlier" />
          <el-option label="缺失" value="missing" />
          <el-option label="聚类" value="cluster" />
          <el-option label="重复" value="duplicate" />
        </el-select>

        <!-- 严重程度 -->
        <el-select
          :model-value="filter.severity"
          placeholder="严重程度"
          clearable
          size="small"
          style="width:110px;"
          @update:model-value="updateFilter('severity', $event)"
        >
          <el-option label="全部" value="" />
          <el-option label="高" value="high" />
          <el-option label="中" value="medium" />
          <el-option label="低" value="low" />
        </el-select>

        <!-- 关键词搜索 -->
        <el-input
          :model-value="filter.keyword"
          placeholder="搜索规则/字段"
          size="small"
          style="width:160px;"
          clearable
          @update:model-value="updateFilter('keyword', $event)"
          @input="applyFilters"
        />

        <el-button size="small" @click="resetFilters">重置</el-button>
      </div>

      <div class="control-right">
        <!-- 数据源切换 -->
        <el-radio-group v-model="dataSource" size="small" @change="onDataSourceChange">
          <el-radio-button value="parquet">📁 样本数据</el-radio-button>
          <el-radio-button value="sql" :disabled="!hasDbConfig">
            🗄️ 源数据
          </el-radio-button>
        </el-radio-group>

        <el-button size="small" type="primary" plain @click="handleExport" :disabled="tableData.length === 0">
          📥 导出 CSV
        </el-button>

        <span class="record-count">共 {{ filteredRecords.length }} 条</span>
      </div>
    </div>

    <!-- ===== 状态栏 ===== -->
    <div class="status-bar">
      <span class="status-item" v-if="dataSource === 'parquet'">
        <span class="label">数据源</span>
        <span class="value">📁 样本 (Parquet)</span>
      </span>
      <span class="status-item" v-else>
        <span class="label">数据源</span>
        <span class="value">🗄️ 全量 (SQL Server)</span>
      </span>
      <span class="status-item">
        <span class="label">已加载</span>
        <span class="value">{{ tableData.length }}</span>
        <span class="unit">行</span>
      </span>
      <span class="status-item" v-if="isStreaming">
        <span class="label">速度</span>
        <span class="value">{{ streamSpeed }}</span>
        <span class="unit">行/秒</span>
      </span>
      <span class="status-item">
        <span class="label">耗时</span>
        <span class="value">{{ elapsedTime }}</span>
        <span class="unit">s</span>
      </span>
      <span class="status-item" v-if="dataSource === 'sql'">
        <span class="label">状态</span>
        <span class="value" :class="{ streaming: isStreaming, complete: isStreamComplete }">
          {{ isStreaming ? '⏳ 加载中...' : isStreamComplete ? '✅ 已完成' : '⏸️ 待加载' }}
        </span>
      </span>
      <span class="status-item" v-if="errorMessage">
        <el-tag type="danger" size="small">{{ errorMessage }}</el-tag>
      </span>
    </div>

    <!-- ===== 数据表格 ===== -->
    <div class="table-wrapper">
      <!-- 空状态：没有诊断记录 -->
      <div v-if="dataSource === 'parquet' && tableData.length === 0 && !isStreaming" class="empty-records">
        <el-empty description="暂无诊断记录，请先在「开始分析」执行场景">
          <el-button type="primary" @click="$emit('go-to-config')">去执行场景</el-button>
        </el-empty>
      </div>

      <!-- 加载中 -->
      <div v-else-if="isStreaming && tableData.length === 0" class="loading-records">
        <el-skeleton :rows="5" animated />
        <div class="loading-status">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>{{ loadingMessage || '正在加载数据...' }}</span>
        </div>
      </div>

      <!-- 数据表格 -->
      <el-table
        v-else
        :data="pagedData"
        border
        size="small"
        max-height="500"
        style="width:100%;"
        v-loading="isStreaming && tableData.length > 0"
      >
        <el-table-column type="index" label="#" width="50" align="center" />
        <el-table-column prop="row" label="行号" width="80" align="center" sortable />
        <el-table-column prop="scenario_name" label="所属场景" min-width="120" sortable />
        <el-table-column prop="record_type_display" label="记录类型" width="100" align="center" sortable>
          <template #default="{ row }">
            <el-tag v-if="row.record_type === 'cluster'" size="small" type="primary">聚类</el-tag>
            <el-tag v-else-if="row.record_type === 'violation'" size="small" type="danger">勾稽</el-tag>
            <el-tag v-else-if="row.record_type === 'outlier'" size="small" type="warning">异常</el-tag>
            <el-tag v-else-if="row.record_type === 'missing'" size="small" type="info">缺失</el-tag>
            <el-tag v-else-if="row.record_type === 'duplicate'" size="small" type="warning">重复</el-tag>
            <el-tag v-else-if="row.record_type === 'entity_concentration'" size="small" type="success">集中</el-tag>
            <el-tag v-else size="small" type="info">其他</el-tag>
          </template>
        </el-table-column>

        <!-- 规则/字段列 -->
        <el-table-column prop="field_display" label="规则/字段" min-width="200" sortable>
          <template #default="{ row }">
            <span v-if="row.record_type === 'cluster'">归属群组</span>
            <span v-else-if="row.record_type === 'violation'">
              <div style="display:flex;flex-direction:column;gap:2px;">
                <span style="font-weight:500;">{{ row.rule || '勾稽规则' }}</span>
                <span v-if="row.values_display" style="color:#909399;font-size:11px;word-break:break-all;">
                  {{ row.values_display }}
                </span>
              </div>
            </span>
            <span v-else>{{ row.field_display || row.field || '—' }}</span>
          </template>
        </el-table-column>

        <el-table-column prop="value_display" label="当前值" min-width="150" sortable>
          <template #default="{ row }">
            <span v-if="row.record_type === 'cluster'">群组 {{ row.cluster_id }}</span>
            <span v-else-if="row.record_type === 'entity_concentration'">{{ row.entity || '—' }}</span>
            <span v-else-if="row.record_type === 'violation' && row.values_display">
              {{ row.values_display }}
            </span>
            <span v-else-if="row.record_type === 'missing' && row.missing_fields">
              {{ row.missing_fields.join(', ') }}
            </span>
            <span v-else>{{ row.value !== undefined && row.value !== null ? row.value : '—' }}</span>
          </template>
        </el-table-column>

        <el-table-column prop="expected" label="预期值/范围" min-width="100">
          <template #default="{ row }">
            <span v-if="row.record_type === 'cluster' || row.record_type === 'entity_concentration' || row.record_type === 'duplicate'">—</span>
            <span v-else-if="row.record_type === 'violation'">相等</span>
            <span v-else>{{ row.expected || '—' }}</span>
          </template>
        </el-table-column>

        <el-table-column prop="deviation" label="偏离程度" width="100" align="center" sortable>
          <template #default="{ row }">
            <span v-if="row.record_type === 'cluster' || row.record_type === 'entity_concentration' || row.record_type === 'duplicate'">—</span>
            <span v-else>{{ row.deviation !== undefined ? row.deviation.toFixed(2) + 'x' : '—' }}</span>
          </template>
        </el-table-column>

        <el-table-column prop="severity" label="严重程度" width="90" align="center">
          <template #default="{ row }">
            <el-tag v-if="row.severity" :type="row.severity === 'high' ? 'danger' : row.severity === 'medium' ? 'warning' : 'info'" size="small">{{ row.severity }}</el-tag>
            <span v-else>—</span>
          </template>
        </el-table-column>

        <!-- 状态列（仅样本模式可编辑） -->
        <el-table-column prop="status" label="状态" width="110" align="center" v-if="dataSource === 'parquet'">
          <template #default="{ row }">
            <el-select :model-value="row.status" size="small" placeholder="状态" @update:model-value="updateRecordStatus(row, $event)">
              <el-option label="待核查" value="pending" />
              <el-option label="已忽略" value="ignored" />
              <el-option label="已处理" value="resolved" />
            </el-select>
          </template>
        </el-table-column>

        <!-- 追溯列 -->
        <el-table-column label="追溯" width="70" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              size="small"
              text
              type="primary"
              @click="handleTrace(row)"
              :disabled="!row.row || !hasDbConfig"
              title="追溯原始数据"
            >
              📦
            </el-button>
          </template>
        </el-table-column>

        <el-table-column label="详情" width="70" align="center" fixed="right">
          <template #default="{ row }">
            <el-button size="small" text type="primary" @click="$emit('show-detail', row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- ===== 分页 ===== -->
    <div class="pagination">
      <el-pagination
        :model-value="currentPage"
        :page-size="pageSize"
        :total="filteredRecords.length"
        :page-sizes="[20, 50, 100, 200]"
        layout="total, sizes, prev, pager, next"
        @update:model-value="currentPage = $event"
        @update:page-size="pageSize = $event"
        @current-change="onPageChange"
        @size-change="onPageChange"
        size="small"
      />
    </div>

    <!-- ===== SQL 显示（调试） ===== -->
    <div v-if="currentSql && dataSource === 'sql'" class="sql-display">
      <el-collapse>
        <el-collapse-item title="📝 查看 SQL" name="sql">
          <pre class="sql-code">{{ currentSql }}</pre>
        </el-collapse-item>
      </el-collapse>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { Loading } from '@element-plus/icons-vue'
import { streamApi } from '../api'

const props = defineProps({
  scenarioResults: { type: Array, default: () => [] },
  allRecords: { type: Array, default: () => [] },
  fieldMapping: { type: Object, default: () => ({}) },
  sessionId: { type: String, default: '' },
  hasDbConfig: { type: Boolean, default: false },
  parquetData: { type: Array, default: () => [] },
  traceContext: { type: Object, default: null }
})

const emit = defineEmits(['show-detail', 'trace', 'update-record-status', 'go-to-config', 'context-change'])

// ===== 状态 =====
const dataSource = ref('parquet')
const tableData = ref([])
const filteredRecords = ref([])
const currentPage = ref(1)
const pageSize = ref(50)

const filter = ref({
  scenario: '',
  keyword: '',
  recordType: '',
  severity: '',
  status: ''
})

const isStreaming = ref(false)
const isStreamComplete = ref(false)
const streamSpeed = ref(0)
const elapsedTime = ref(0)
const errorMessage = ref('')
const currentSql = ref('')
const loadingMessage = ref('')

let startTime = null
let lastRowTime = null
let speedSamples = []
let abortController = null
let rowCount = 0

// ===== 计算属性 =====
const pagedData = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return filteredRecords.value.slice(start, end)
})

// ===== 更新筛选条件 =====
function updateFilter(key, value) {
  filter.value[key] = value
  applyFilters()
}

function applyFilters() {
  let records = [...tableData.value]
  const f = filter.value

  if (f.scenario) {
    records = records.filter(r => String(r.scenario_id) === String(f.scenario))
  }
  if (f.keyword) {
    const kw = f.keyword.toLowerCase()
    records = records.filter(r =>
      (r.field && String(r.field).toLowerCase().includes(kw)) ||
      (r.field_display && String(r.field_display).toLowerCase().includes(kw)) ||
      (r.rule && String(r.rule).toLowerCase().includes(kw)) ||
      (r.entity && String(r.entity).toLowerCase().includes(kw)) ||
      (r.values_display && String(r.values_display).toLowerCase().includes(kw))
    )
  }
  if (f.recordType) {
    records = records.filter(r => r.record_type === f.recordType)
  }
  if (f.severity) {
    records = records.filter(r => r.severity === f.severity)
  }
  if (f.status) {
    records = records.filter(r => r.status === f.status)
  }

  filteredRecords.value = records
  currentPage.value = 1
}

function resetFilters() {
  filter.value = { scenario: '', keyword: '', recordType: '', severity: '', status: '' }
  applyFilters()
}

// ===== 数据源切换 =====
function onDataSourceChange(value) {
  if (value === 'parquet') {
    loadParquetData()
  } else if (value === 'sql') {
    if (!props.hasDbConfig) {
      ElMessage.warning('当前会话没有关联的数据库配置，请使用数据库方式加载数据')
      dataSource.value = 'parquet'
      return
    }
    loadSqlData()
  }
}

// ===== 加载 Parquet 数据 =====
function loadParquetData() {
  if (abortController) {
    abortController.abort()
    abortController = null
  }
  isStreaming.value = false
  isStreamComplete.value = true
  errorMessage.value = ''
  loadingMessage.value = ''

  const records = props.allRecords || []
  if (records.length === 0) {
    tableData.value = []
    filteredRecords.value = []
    rowCount = 0
    elapsedTime.value = 0
    return
  }

  tableData.value = records.map(r => ({
    ...r,
    field_display: r.field_display || r.field || '—',
    record_type_display: r.record_type_display || '其他',
    scenario_name: r.scenario_name || '未知场景'
  }))

  rowCount = tableData.value.length
  elapsedTime.value = 0
  applyFilters()
}

// ===== 加载 SQL 数据 =====
function loadSqlData() {
  if (!props.traceContext) {
    ElMessage.warning('没有追溯上下文，请从分析摘要点击规则/公司/字段名称')
    dataSource.value = 'parquet'
    return
  }

  // 重置状态
  tableData.value = []
  filteredRecords.value = []
  rowCount = 0
  isStreaming.value = true
  isStreamComplete.value = false
  errorMessage.value = ''
  currentSql.value = ''
  streamSpeed.value = 0
  elapsedTime.value = 0
  speedSamples = []
  loadingMessage.value = '正在连接数据库...'
  startTime = Date.now()
  lastRowTime = Date.now()

  abortController = new AbortController()

  const context = props.traceContext

  streamApi.streamQuery({
    sessionId: props.sessionId,
    context: context,
    batchSize: 100,
    maxRows: 10000,
    abortController: abortController,
    onInfo: (description, sql) => {
      currentSql.value = sql
      loadingMessage.value = '正在执行查询...'
      console.log('[SQL追溯] SQL:', sql)
    },
    onMeta: (columns) => {
      loadingMessage.value = '已连接，开始接收数据...'
      console.log('[SQL追溯] 列信息:', columns)
    },
    onChunk: (row, count) => {
      const record = {
        ...row,
        row: row.id,
        record_type: 'full_data',
        record_type_display: '全量数据',
        scenario_name: '源数据追溯',
        field_display: '—',
        value_display: '—',
        status: 'pending'
      }
      tableData.value.push(record)
      rowCount = count

      const now = Date.now()
      const delta = (now - lastRowTime.value) / 1000
      if (delta > 0) {
        const currentSpeed = 1 / delta
        speedSamples.push(currentSpeed)
        if (speedSamples.length > 10) speedSamples.shift()
        streamSpeed.value = Math.round(
          speedSamples.reduce((a, b) => a + b, 0) / speedSamples.length
        )
      }
      lastRowTime.value = now
      elapsedTime.value = (now - startTime.value) / 1000
      loadingMessage.value = `已加载 ${rowCount} 行`

      applyFilters()
    },
    onComplete: (count, cancelled) => {
      isStreaming.value = false
      isStreamComplete.value = true
      elapsedTime.value = (Date.now() - startTime.value) / 1000

      if (cancelled) {
        loadingMessage.value = '已停止加载'
        ElMessage.info('已停止加载')
      } else {
        loadingMessage.value = `加载完成，共 ${count} 行`
        if (count > 0) {
          ElMessage.success(`已加载 ${count} 行数据`)
          applyFilters()
        } else {
          ElMessage.info('查询结果为空，没有找到匹配的记录')
        }
      }
    },
    onError: (message, isWarning) => {
      isStreaming.value = false
      if (isWarning) {
        ElMessage.warning(message)
      } else {
        errorMessage.value = message
        ElMessage.error(message)
        loadingMessage.value = '加载失败: ' + message
      }
    }
  })
}

// ===== 处理追溯 =====
function handleTrace(row) {
  if (!row.row) {
    ElMessage.warning('该记录没有行号信息')
    return
  }

  if (!props.hasDbConfig) {
    ElMessage.warning('没有数据库配置，无法追溯全量数据')
    return
  }

  // 触发追溯事件，切换到 SQL 模式
  emit('trace', {
    type: 'row',
    data: row.row,
    context: { row_ids: [row.row] }
  })

  // 切换到 SQL 数据源
  dataSource.value = 'sql'
  setTimeout(() => loadSqlData(), 300)
}

// ===== 更新记录状态 =====
function updateRecordStatus(row, status) {
  row.status = status
  emit('update-record-status', row, status)
}

// ===== 导出 =====
function handleExport() {
  if (filteredRecords.value.length === 0) {
    ElMessage.warning('没有数据可导出')
    return
  }

  const headers = ['行号', '所属场景', '记录类型', '规则/字段', '当前值', '预期值', '偏离程度', '严重程度', '状态']
  const rows = filteredRecords.value.map(r => [
    r.row || '',
    r.scenario_name || '',
    r.record_type_display || '其他',
    r.record_type === 'violation' ? (r.rule || '') : (r.field_display || r.field || ''),
    r.values_display || r.value_display || '',
    r.record_type === 'violation' ? '相等' : (r.expected || ''),
    r.record_type === 'violation' ? (r.diff !== undefined ? r.diff.toFixed(4) : '') : (r.deviation !== undefined ? r.deviation.toFixed(2) : ''),
    r.severity || '',
    { pending: '待核查', ignored: '已忽略', resolved: '已处理' }[r.status] || r.status || ''
  ])

  const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
  const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `数据追溯_${dataSource.value}_${new Date().toISOString().slice(0, 10)}.csv`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  ElMessage.success('导出成功')
}

function onPageChange() {}

// ===== 监听上下文变化 =====
watch(() => props.traceContext, (newContext) => {
  if (newContext && Object.keys(newContext).length > 0 && props.hasDbConfig) {
    dataSource.value = 'sql'
    setTimeout(() => loadSqlData(), 300)
  }
}, { deep: true })

// ===== 监听场景结果变化 =====
watch(() => props.allRecords, (newRecords) => {
  if (dataSource.value === 'parquet' && newRecords && newRecords.length > 0) {
    loadParquetData()
  }
}, { deep: true })

// ===== 生命周期 =====
onMounted(() => {
  loadParquetData()
})

onUnmounted(() => {
  if (abortController) {
    abortController.abort()
    abortController = null
  }
})

defineExpose({
  loadParquetData,
  loadSqlData,
  applyFilters,
  resetFilters,
  dataSource
})
</script>

<style scoped>
.data-decode-trace {
  padding: 4px 0;
}

.control-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  padding: 10px 14px;
  background: #f5f7fa;
  border-radius: 8px;
  margin-bottom: 10px;
}

.control-left {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.control-right {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.record-count {
  font-size: 12px;
  color: #909399;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 4px 12px;
  background: #fafafa;
  border-radius: 6px;
  flex-shrink: 0;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.status-item {
  display: flex;
  align-items: baseline;
  gap: 4px;
  font-size: 12px;
}

.status-item .label {
  color: #909399;
}

.status-item .value {
  font-weight: 600;
  color: #2c3e50;
}

.status-item .value.streaming {
  color: #e6a23c;
}

.status-item .value.complete {
  color: #67c23a;
}

.status-item .unit {
  color: #909399;
  font-size: 11px;
}

.table-wrapper {
  overflow-x: auto;
  margin-bottom: 12px;
  min-height: 200px;
}

.empty-records {
  padding: 40px 0;
}

.loading-records {
  padding: 20px 0;
}

.loading-status {
  margin-top: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
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

.pagination {
  display: flex;
  justify-content: flex-end;
}

.sql-display {
  margin-top: 8px;
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

@media (max-width: 768px) {
  .control-bar {
    flex-direction: column;
    align-items: stretch;
  }
  .control-left {
    flex-wrap: wrap;
  }
  .control-right {
    justify-content: flex-start;
  }
  .status-bar {
    gap: 10px;
    padding: 8px 10px;
  }
}
</style>