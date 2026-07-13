<template>
  <el-dialog
    v-model="visible"
    :title="dialogTitle"
    width="90%"
    top="5vh"
    destroy-on-close
    :close-on-click-modal="true"
    @close="handleClose"
  >
    <div class="preview-dialog-body">
      <!-- 统计信息 -->
      <div class="preview-stats">
        <el-tag type="info" size="small">共 {{ totalRows }} 行</el-tag>
        <el-tag type="success" size="small">当前显示 {{ tableData.length }} 行</el-tag>
        <el-tag type="warning" size="small" v-if="filterDesc">筛选: {{ filterDesc }}</el-tag>
        <el-button size="small" type="primary" plain @click="handleExport">📥 导出 CSV</el-button>
      </div>

      <!-- 数据表格 -->
      <div class="preview-table-wrapper">
        <el-table
          :data="tableData"
          border
          size="small"
          max-height="500"
          style="width: 100%"
          v-loading="loading"
          row-key="__row_index"
        >
          <el-table-column type="index" label="#" width="50" align="center" />
          <el-table-column
            v-for="col in tableColumns"
            :key="col"
            :prop="col"
            :label="col"
            min-width="100"
            show-overflow-tooltip
          >
            <template #default="scope">
              <span v-if="scope.row[col] === null || scope.row[col] === undefined" style="color: #ccc;">(空)</span>
              <span v-else-if="typeof scope.row[col] === 'number'">{{ scope.row[col].toFixed(4) }}</span>
              <span v-else>{{ scope.row[col] }}</span>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <!-- 分页 -->
      <div class="preview-pagination">
        <el-pagination
          v-model:page-size="pageSize"
          v-model:current-page="currentPage"
          :total="totalRows"
          :page-sizes="[50, 100, 200, 500]"
          layout="total, sizes, prev, pager, next"
          @size-change="onPageChange"
          @current-change="onPageChange"
          size="small"
        />
      </div>
    </div>
  </el-dialog>
</template>

<script setup>
import { ref, watch, computed, toRefs } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false
  },
  sessionId: {
    type: String,
    default: ''
  },
  title: {
    type: String,
    default: '数据预览'
  },
  filters: {
    type: Array,
    default: () => []
  },
  fields: {
    type: Array,
    default: null
  }
})

const emit = defineEmits(['update:modelValue', 'close'])

// ===== 状态 =====
const visible = ref(false)
const loading = ref(false)
const tableData = ref([])
const tableColumns = ref([])
const totalRows = ref(0)
const filterDesc = ref('')
const currentPage = ref(1)
const pageSize = ref(100)

// 使用 toRefs 响应式监听 props
const { modelValue, sessionId, title, filters, fields } = toRefs(props)

// ===== 计算属性 =====
const dialogTitle = computed(() => {
  const extra = filters.value.length > 0 ? ` (${filters.value.length} 个条件)` : ''
  return `🔍 ${title.value}${extra}`
})

// ===== 监听弹窗显示 =====
watch(() => props.modelValue, (val) => {
  visible.value = val
  if (val) {
    currentPage.value = 1
    loadData()
  }
})

watch(visible, (val) => {
  emit('update:modelValue', val)
  if (!val) {
    emit('close')
  }
})

// ===== 加载数据 =====
async function loadData() {
  if (!props.sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }

  if (loading.value) return

  loading.value = true
  try {
    const response = await api.post('/data/preview', {
      session_id: props.sessionId,
      filters: props.filters.map(f => ({
        field: f.field,
        condition: f.condition,
        value: f.value
      })),
      fields: props.fields,
      page: currentPage.value,
      page_size: pageSize.value
    })

    const rows = (response.rows || []).map((row, index) => ({
      ...row,
      __row_index: (currentPage.value - 1) * pageSize.value + index
    }))

    tableData.value = rows
    tableColumns.value = response.columns || []
    totalRows.value = response.total || 0
    filterDesc.value = response.filter_desc || ''
  } catch (err) {
    ElMessage.error('加载数据失败: ' + (err.message || '未知错误'))
    tableData.value = []
    tableColumns.value = []
    totalRows.value = 0
  } finally {
    loading.value = false
  }
}

// ===== 分页变化 =====
function onPageChange() {
  loadData()
}

// ===== 导出 CSV =====
function handleExport() {
  if (tableData.value.length === 0) {
    ElMessage.warning('没有数据可导出')
    return
  }

  const header = tableColumns.value.join(',')
  const body = tableData.value.map(row => {
    return tableColumns.value.map(col => {
      const val = row[col]
      if (val === null || val === undefined) return ''
      if (typeof val === 'string' && val.includes(',')) {
        return `"${val}"`
      }
      return val
    }).join(',')
  }).join('\n')

  const csv = header + '\n' + body
  const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `data_export_${new Date().toISOString().slice(0, 10)}.csv`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  ElMessage.success('导出成功')
}

// ===== 关闭 =====
function handleClose() {
  visible.value = false
  emit('update:modelValue', false)
  emit('close')
}
</script>

<style scoped>
.preview-dialog-body {
  padding: 4px 0;
}

.preview-stats {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
  padding: 8px 12px;
  background: #f5f7fa;
  border-radius: 8px;
}

.preview-table-wrapper {
  overflow-x: auto;
}

.preview-pagination {
  margin-top: 16px;
  display: flex;
  justify-content: flex-end;
}
</style>