<template>
  <div class="data-overview">
    <h2>📊 数据概览</h2>
    <p class="subtitle">查看当前表的字段分布和统计特征，以及数据质量诊断</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="overview-content">
      <!-- ===== 表选择器 ===== -->
      <TableSelector
        v-model="currentTable"
        :table-names="tableNames"
        :is-multi-table="isMultiTable"
        @change="onTableChange"
      />

      <!-- ===== 诊断卡片（新增） ===== -->
      <div class="diagnosis-cards">
        <!-- 常量列 -->
        <el-card v-if="constantColumns.length > 0" shadow="hover" class="diagnosis-card danger">
          <div class="card-icon">📌</div>
          <div class="card-content">
            <div class="card-title">发现 {{ constantColumns.length }} 列常量列</div>
            <div class="card-desc">这些列所有值完全相同，不携带任何信息，建议剔除</div>
            <div class="card-tags">
              <el-tag v-for="col in constantColumns.slice(0, 5)" :key="col" size="small" type="info">{{ col }}</el-tag>
              <el-tag v-if="constantColumns.length > 5" size="small" type="info">+{{ constantColumns.length - 5 }}</el-tag>
            </div>
          </div>
        </el-card>

        <!-- 全空列 -->
        <el-card v-if="emptyColumns.length > 0" shadow="hover" class="diagnosis-card danger">
          <div class="card-icon">🗑️</div>
          <div class="card-content">
            <div class="card-title">发现 {{ emptyColumns.length }} 列全空列</div>
            <div class="card-desc">这些列全部为空值，不携带任何信息，建议直接删除</div>
            <div class="card-tags">
              <el-tag v-for="col in emptyColumns.slice(0, 5)" :key="col" size="small" type="info">{{ col }}</el-tag>
              <el-tag v-if="emptyColumns.length > 5" size="small" type="info">+{{ emptyColumns.length - 5 }}</el-tag>
            </div>
          </div>
        </el-card>

        <!-- 单位混用 -->
        <el-card v-if="unitMismatch" shadow="hover" class="diagnosis-card danger">
          <div class="card-icon">⚠️</div>
          <div class="card-content">
            <div class="card-title">单位混用：{{ unitField }} 存在 {{ unitCount }} 种单位</div>
            <div class="card-desc">跨公司数值不可直接比较，建议统一为"元"</div>
            <div class="card-tags">
              <el-tag v-for="(label, code) in unitMap" :key="code" size="small" type="warning">{{ code }} = {{ label }}</el-tag>
            </div>
          </div>
        </el-card>

        <!-- 字段命名 -->
        <el-card v-if="namingScore < 0.5" shadow="hover" class="diagnosis-card warning">
          <div class="card-icon">📝</div>
          <div class="card-content">
            <div class="card-title">字段命名可读性差：{{ namingScore * 100 }}% 为数字后缀</div>
            <div class="card-desc">建议在「场景分析」中配置字段映射，提升可读性</div>
          </div>
        </el-card>

        <!-- 数据年龄 -->
        <el-card v-if="dataAge" shadow="hover" class="diagnosis-card info">
          <div class="card-icon">📅</div>
          <div class="card-content">
            <div class="card-title">数据为 {{ dataAge.year }} 年快照</div>
            <div class="card-desc">导入窗口：{{ dataAge.importStart }} ~ {{ dataAge.importEnd }}，不支持跨年趋势分析</div>
          </div>
        </el-card>

        <!-- 列保留阈值 -->
        <el-card v-if="columnRetention" shadow="hover" class="diagnosis-card info">
          <div class="card-icon">📊</div>
          <div class="card-content">
            <div class="card-title">列保留建议：缺失率 &lt; 50% 可保留 {{ columnRetention.keep }} 列</div>
            <div class="card-desc">覆盖率 {{ columnRetention.coverage }}%，建议在分析前剔除高缺失列</div>
          </div>
        </el-card>

        <!-- 采样偏差 -->
        <el-card v-if="sampleBias" shadow="hover" class="diagnosis-card warning">
          <div class="card-icon">🎯</div>
          <div class="card-content">
            <div class="card-title">头部 {{ sampleBias.topPct }}% 公司贡献 {{ sampleBias.topRecords }}% 记录</div>
            <div class="card-desc">数据存在严重采样偏差，结论仅适用于大型企业</div>
          </div>
        </el-card>

        <!-- 非稳健统计 -->
        <el-card v-if="unstableStats.length > 0" shadow="hover" class="diagnosis-card warning">
          <div class="card-icon">📈</div>
          <div class="card-content">
            <div class="card-title">{{ unstableStats.length }} 个字段均值被极端值绑架</div>
            <div class="card-desc">建议使用中位数描述中心趋势</div>
            <div class="card-tags">
              <el-tag v-for="item in unstableStats.slice(0, 3)" :key="item.field" size="small" type="warning">
                {{ item.field }} (均值/中位数 = {{ item.ratio }})
              </el-tag>
              <el-tag v-if="unstableStats.length > 3" size="small" type="info">+{{ unstableStats.length - 3 }}</el-tag>
            </div>
          </div>
        </el-card>
      </div>

      <!-- 统计卡片 -->
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-value">{{ dataShape.rows || 0 }}</div>
          <div class="stat-label">总行数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ dataShape.columns || 0 }}</div>
          <div class="stat-label">总列数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ variableTypesCount }}</div>
          <div class="stat-label">变量类型</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ missingFieldsCount }}</div>
          <div class="stat-label">缺失字段</div>
        </div>
      </div>

      <!-- 图表区域 -->
      <div class="charts-row">
        <div class="chart-card">
          <div class="chart-header">
            <span class="chart-title">📊 变量类型分布</span>
          </div>
          <v-chart v-if="hasPieData" :option="pieOption" class="chart-container" />
          <div v-else class="chart-empty">暂无变量类型数据</div>
        </div>

        <div class="chart-card">
          <div class="chart-header">
            <span class="chart-title">📊 分类变量类别数</span>
          </div>
          <v-chart v-if="hasCategoryCountData" :key="'cat_count_' + catCountKey" :option="categoryCountOption" class="chart-container" />
          <div v-else class="chart-empty">暂无分类变量数据</div>
        </div>
      </div>

      <div class="chart-card full-width">
        <div class="chart-header">
          <span class="chart-title">📊 连续变量取值范围</span>
        </div>
        <v-chart v-if="hasContinuousRangeData" :key="'cont_range_' + contRangeKey" :option="continuousRangeOption" class="chart-container" style="height: 280px;" />
        <div v-else class="chart-empty">暂无连续变量数据</div>
      </div>

      <!-- 字段详情 -->
      <div class="section">
        <h4>📋 字段详情</h4>
        <el-tabs v-model="fieldTab" class="field-tabs">
          <el-tab-pane label="连续变量" name="continuous">
            <el-table :data="continuousVarList" border size="small" max-height="420" style="width: 100%;">
              <el-table-column prop="name" label="字段名" width="120" fixed="left">
                <template #default="{ row }">
                  <span class="field-name-link" @click="openFieldDetail(row.name)">{{ row.name }}</span>
                </template>
              </el-table-column>
              <el-table-column prop="count" label="样本量" width="80" align="center">
                <template #default="{ row }">
                  <span class="field-name-link" @click="showNonMissingRows(row.name)">
                    {{ row.count }}
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="missing_pct" label="缺失率" width="80" align="center">
                <template #default="{ row }">
                  <span
                    class="field-name-link"
                    :style="{ color: row.missing_pct > 20 ? '#F56C6C' : '#909399' }"
                    @click="showMissingRows(row.name)"
                  >
                    {{ row.missing_pct.toFixed(1) }}%
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="mean" label="均值" width="100" align="center">
                <template #default="{ row }">
                  {{ row.mean !== undefined && row.mean !== null ? Number(row.mean).toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="median" label="中位数" width="100" align="center">
                <template #default="{ row }">
                  {{ row.median !== undefined && row.median !== null ? Number(row.median).toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="std" label="标准差" width="100" align="center">
                <template #default="{ row }">
                  {{ row.std !== undefined && row.std !== null ? Number(row.std).toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="min" label="最小值" width="90" align="center">
                <template #default="{ row }">
                  {{ row.min !== undefined && row.min !== null ? Number(row.min).toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="max" label="最大值" width="90" align="center">
                <template #default="{ row }">
                  {{ row.max !== undefined && row.max !== null ? Number(row.max).toFixed(2) : '-' }}
                </template>
              </el-table-column>
              <el-table-column label="分布区间" min-width="160" align="center">
                <template #default="{ row }">
                  <div class="range-bar-wrapper" :title="`${Number(row.min).toFixed(2)} ~ ${Number(row.max).toFixed(2)}，均值 ${Number(row.mean).toFixed(2)}`">
                    <div class="range-bar-track">
                      <div class="range-bar-fill" :style="{ left: getRangeBarLeft(row), width: getRangeBarWidth(row) }" />
                      <div class="range-bar-dot" :style="{ left: getRangeBarDot(row) }" />
                    </div>
                    <div class="range-bar-labels">
                      <span class="range-min">{{ row.min !== undefined && row.min !== null ? Number(row.min).toFixed(1) : '-' }}</span>
                      <span class="range-max">{{ row.max !== undefined && row.max !== null ? Number(row.max).toFixed(1) : '-' }}</span>
                    </div>
                  </div>
                </template>
              </el-table-column>
            </el-table>
            <div v-if="continuousVarList.length === 0" class="empty-tip">暂无连续变量</div>
          </el-tab-pane>

          <el-tab-pane label="分类变量" name="categorical">
            <el-table :data="categoricalVarList" border size="small" max-height="420" style="width: 100%;">
              <el-table-column prop="name" label="字段名" width="120" fixed="left">
                <template #default="{ row }">
                  <span class="field-name-link" @click="openFieldDetail(row.name)">{{ row.name }}</span>
                </template>
              </el-table-column>
              <el-table-column prop="count" label="样本量" width="80" align="center">
                <template #default="{ row }">
                  <span class="field-name-link" @click="showNonMissingRows(row.name)">
                    {{ row.count }}
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="missing_pct" label="缺失率" width="80" align="center">
                <template #default="{ row }">
                  <span :style="{ color: row.missing_pct > 20 ? '#F56C6C' : '#909399' }" class="field-name-link" @click="showMissingRows(row.name)">
                    {{ row.missing_pct.toFixed(1) }}%
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="n_unique" label="类别数" width="80" align="center" />
              <el-table-column prop="mode" label="众数" width="100" align="center" show-overflow-tooltip />
              <el-table-column prop="mode_pct" label="众数占比" width="90" align="center">
                <template #default="{ row }">
                  {{ row.mode_pct !== undefined ? Number(row.mode_pct).toFixed(1) + '%' : '-' }}
                </template>
              </el-table-column>
              <el-table-column label="类别分布 (Top3)" min-width="180">
                <template #default="{ row }">
                  <div v-if="row.topCategories && row.topCategories.length > 0" class="category-bars">
                    <div v-for="(item, idx) in row.topCategories.slice(0, 3)" :key="idx" class="category-bar-item">
                      <span class="category-name">{{ item.name }}</span>
                      <div class="category-bar-track">
                        <div class="category-bar-fill" :style="{ width: item.pct + '%', backgroundColor: getCategoryColor(idx) }" />
                      </div>
                      <span class="category-pct">{{ item.pct.toFixed(1) }}%</span>
                    </div>
                  </div>
                  <span v-else class="text-muted">-</span>
                </template>
              </el-table-column>
            </el-table>
            <div v-if="categoricalVarList.length === 0" class="empty-tip">暂无分类变量</div>
          </el-tab-pane>

          <el-tab-pane label="其他类型" name="other">
            <el-table :data="otherVarList" border size="small" max-height="420" style="width: 100%;">
              <el-table-column prop="name" label="字段名" width="140" fixed="left">
                <template #default="{ row }">
                  <span class="field-name-link" @click="openFieldDetail(row.name)">{{ row.name }}</span>
                </template>
              </el-table-column>
              <el-table-column prop="type_desc" label="类型" width="100" align="center" />
              <el-table-column prop="count" label="样本量" width="80" align="center">
                <template #default="{ row }">
                  <span class="field-name-link" @click="showNonMissingRows(row.name)">
                    {{ row.count }}
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="missing_pct" label="缺失率" width="80" align="center">
                <template #default="{ row }">
                  <span :style="{ color: row.missing_pct > 20 ? '#F56C6C' : '#909399' }" class="field-name-link" @click="showMissingRows(row.name)">
                    {{ row.missing_pct.toFixed(1) }}%
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="key_info" label="关键信息" min-width="200" show-overflow-tooltip />
            </el-table>
            <div v-if="otherVarList.length === 0" class="empty-tip">暂无其他类型变量</div>
          </el-tab-pane>
        </el-tabs>
      </div>
    </div>

    <div v-else-if="!loading" class="empty-state">
      <el-empty description="请先上传数据并完成分析">
        <el-button type="primary" @click="goToUpload">去上传数据</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { useFieldDetailStore } from '../stores/fieldDetail'
import { reportApi } from '../api/report'
import { openDataPreview } from '../components/DataPreviewDialog'
import TableSelector from '../components/TableSelector.vue'

const router = useRouter()
const sessionStore = useSessionStore()
const fieldDetailStore = useFieldDetailStore()

const loading = ref(false)
const reportData = ref(null)
const fieldTab = ref('continuous')
const catCountKey = ref(0)
const contRangeKey = ref(0)

// ===== 表选择器状态 =====
const currentTable = ref('merged')
const tableNames = ref([])
const isMultiTable = ref(false)

const typeDisplay = {
  continuous: '连续变量',
  categorical: '分类变量',
  categorical_numeric: '数值型分类',
  ordinal: '有序分类',
  datetime: '日期时间',
  identifier: '标识符',
  text: '文本'
}

// ===== 当前数据（从 all_tables 取） =====
const allTables = computed(() => reportData.value?.all_tables || {})
const currentData = computed(() => {
  if (!allTables.value) return {}
  const target = currentTable.value
  return allTables.value[target] || allTables.value['merged'] || {}
})

const dataShape = computed(() => currentData.value?.data_shape || { rows: 0, columns: 0 })
const variableTypes = computed(() => currentData.value?.variable_types || {})
const summaries = computed(() => currentData.value?.variable_summaries || {})
const quality = computed(() => currentData.value?.quality_report || {})
const correlations = computed(() => currentData.value?.correlations || {})
const columnNames = computed(() => currentData.value?.column_names || [])

// ===== 诊断卡片数据（新增） =====
const constantColumns = computed(() => {
  const cols = []
  for (const [col, info] of Object.entries(summaries.value)) {
    if (info.n_unique === 1) {
      cols.push(col)
    }
  }
  return cols
})

const emptyColumns = computed(() => {
  const cols = []
  for (const [col, info] of Object.entries(summaries.value)) {
    if (info.missing_pct === 100) {
      cols.push(col)
    }
  }
  return cols
})

const unitMismatch = computed(() => {
  // 检测 cunit 字段是否存在多单位
  const cunitInfo = summaries.value.cunit
  if (!cunitInfo) return false
  const topCats = cunitInfo.top_categories || {}
  if (Object.keys(topCats).length > 1) {
    return true
  }
  return false
})

const unitField = computed(() => {
  if (unitMismatch.value) return 'cunit'
  return ''
})

const unitCount = computed(() => {
  const cunitInfo = summaries.value.cunit
  if (!cunitInfo) return 0
  const topCats = cunitInfo.top_categories || {}
  return Object.keys(topCats).length
})

const unitMap = computed(() => {
  // 从 cunit 的 top_categories 中提取
  const cunitInfo = summaries.value.cunit
  if (!cunitInfo) return {}
  const topCats = cunitInfo.top_categories || {}
  const map = {}
  // 根据实际数据推断单位含义
  const keys = Object.keys(topCats)
  const labels = ['元', '千元', '万元', '十万元', '百万元']
  keys.forEach((k, idx) => {
    map[k] = labels[idx] || '未知'
  })
  return map
})

const namingScore = computed(() => {
  if (!columnNames.value || columnNames.value.length === 0) return 1
  const numericSuffix = columnNames.value.filter(col => /^companyfixasset\d+$/.test(col))
  return numericSuffix.length / columnNames.value.length
})

const dataAge = computed(() => {
  const declaredate = summaries.value.declaredate
  const reportdate = summaries.value.reportdate
  if (!declaredate || !reportdate) return null
  const minDate = declaredate.min_date
  const maxDate = declaredate.max_date
  if (!minDate || !maxDate) return null
  // 检查 reportdate 是否集中在某一年
  const yearInfo = summaries.value.reportdate_year
  if (yearInfo && yearInfo.top_categories) {
    const topYear = Object.keys(yearInfo.top_categories)[0]
    return {
      year: topYear,
      importStart: minDate.split(' ')[0],
      importEnd: maxDate.split(' ')[0]
    }
  }
  return null
})

const columnRetention = computed(() => {
  const missingList = quality.value?.missing || []
  if (!missingList || missingList.length === 0) return null
  const totalCols = columnNames.value.length || 1
  const keepCols = missingList.filter(m => m.percent < 50).length
  const coverage = Math.round((keepCols / totalCols) * 100)
  return {
    keep: keepCols,
    coverage: coverage
  }
})

const sampleBias = computed(() => {
  // 从 companycode 频次计算
  const codeSummary = summaries.value.companycode
  if (!codeSummary) return null
  const total = codeSummary.count || 0
  // 简化：假设 top_categories 存在，计算前 5% 的占比
  const topCats = codeSummary.top_categories || {}
  const entries = Object.entries(topCats)
  if (entries.length === 0) return null
  const sorted = entries.sort((a, b) => b[1] - a[1])
  const top5Pct = Math.max(1, Math.round(sorted.length * 0.05))
  const topRecords = sorted.slice(0, top5Pct).reduce((sum, item) => sum + item[1], 0)
  const pct = Math.round((topRecords / total) * 100)
  return {
    topPct: top5Pct / sorted.length * 100,
    topRecords: pct
  }
})

const unstableStats = computed(() => {
  const result = []
  for (const [field, info] of Object.entries(summaries.value)) {
    if (info.type === 'continuous') {
      const mean = info.mean
      const median = info.median
      if (mean !== undefined && median !== undefined && mean !== 0 && median !== 0) {
        const ratio = Math.abs(mean / median)
        if (ratio > 5) {
          result.push({
            field,
            ratio: ratio.toFixed(1)
          })
        }
      }
    }
  }
  return result
})

// ===== 其他原有逻辑保持不变 =====
const variableTypesCount = computed(() => Object.keys(typeCounts.value).length)
const missingFieldsCount = computed(() => quality.value?.missing?.length || 0)

const typeCounts = computed(() => {
  const counts = {}
  Object.values(variableTypes.value).forEach(info => {
    const typ = info.type || 'unknown'
    counts[typ] = (counts[typ] || 0) + 1
  })
  return counts
})

const continuousVarList = computed(() => {
  const result = []
  Object.entries(summaries.value).forEach(([name, info]) => {
    if (info.type === 'continuous') {
      const minVal = info.min !== undefined && info.min !== null ? Number(info.min) : 0
      const maxVal = info.max !== undefined && info.max !== null ? Number(info.max) : 0
      const meanVal = info.mean !== undefined && info.mean !== null ? Number(info.mean) : 0
      result.push({
        name,
        count: info.count || 0,
        missing_pct: info.missing_pct || 0,
        mean: meanVal,
        median: info.median !== undefined && info.median !== null ? Number(info.median) : 0,
        std: info.std !== undefined && info.std !== null ? Number(info.std) : 0,
        min: minVal,
        max: maxVal,
        _min: minVal,
        _max: maxVal,
        _mean: meanVal
      })
    }
  })
  result.sort((a, b) => b.missing_pct - a.missing_pct)
  return result
})

const categoricalVarList = computed(() => {
  const result = []
  Object.entries(summaries.value).forEach(([name, info]) => {
    const typ = info.type
    if (typ === 'categorical' || typ === 'categorical_numeric' || typ === 'ordinal') {
      let topCats = []
      const total = info.count || 0
      if (info.top_categories && Object.keys(info.top_categories).length > 0) {
        const entries = Object.entries(info.top_categories)
        const totalCount = entries.reduce((s, e) => s + e[1], 0)
        topCats = entries.map(([cat, count]) => ({
          name: String(cat),
          count: count,
          pct: totalCount > 0 ? (count / totalCount * 100) : 0
        })).sort((a, b) => b.count - a.count)
      } else if (info.value_counts && Object.keys(info.value_counts).length > 0) {
        const entries = Object.entries(info.value_counts)
        const totalCount = entries.reduce((s, e) => s + e[1], 0)
        topCats = entries.map(([cat, count]) => ({
          name: String(cat),
          count: count,
          pct: totalCount > 0 ? (count / totalCount * 100) : 0
        })).sort((a, b) => b.count - a.count)
      }
      result.push({
        name,
        count: info.count || 0,
        missing_pct: info.missing_pct || 0,
        n_unique: info.n_unique || 0,
        mode: info.mode || '-',
        mode_pct: info.mode_pct || 0,
        topCategories: topCats.slice(0, 5)
      })
    }
  })
  result.sort((a, b) => b.missing_pct - a.missing_pct)
  return result
})

const otherVarList = computed(() => {
  const result = []
  Object.entries(summaries.value).forEach(([name, info]) => {
    const typ = info.type
    if (typ === 'datetime' || typ === 'identifier' || typ === 'text') {
      let keyInfo = ''
      if (typ === 'datetime') {
        const minDate = info.min_date || info.min
        const maxDate = info.max_date || info.max
        if (minDate && maxDate) {
          keyInfo = `范围: ${minDate} ~ ${maxDate}`
        } else {
          keyInfo = `唯一值: ${info.n_unique || 0}`
        }
      } else if (typ === 'identifier') {
        keyInfo = `唯一值: ${info.n_unique || 0}`
      } else if (typ === 'text') {
        keyInfo = `唯一值: ${info.n_unique || 0}`
      }
      result.push({
        name,
        type_desc: info.type_desc || typ,
        count: info.count || 0,
        missing_pct: info.missing_pct || 0,
        key_info: keyInfo
      })
    }
  })
  return result
})

// ===== 图表相关 =====
const hasPieData = computed(() => Object.keys(typeCounts.value).length > 0)
const pieOption = computed(() => {
  const counts = typeCounts.value
  const colorMap = {
    continuous: '#409EFF', categorical: '#67C23A',
    categorical_numeric: '#E6A23C', ordinal: '#F56C6C',
    datetime: '#9B59B6', identifier: '#1ABC9C', text: '#95A5A6'
  }
  const data = Object.entries(counts).map(([type, count]) => ({
    name: typeDisplay[type] || type,
    value: count,
    itemStyle: { color: colorMap[type] || '#909399' }
  }))
  return {
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { orient: 'vertical', left: 'left', top: 'center', itemWidth: 12, itemHeight: 12 },
    series: [{
      type: 'pie', radius: ['40%', '65%'], center: ['55%', '50%'],
      avoidLabelOverlap: true, label: { show: true, formatter: '{d}%', fontSize: 11 },
      labelLine: { show: true }, emphasis: { scale: true }, data
    }]
  }
})

const derivedPatterns = ['_year', '_month', '_quarter', '_week', '_weekday', '_day', '_is_weekend']
const isDerivedField = (name) => derivedPatterns.some(p => name.endsWith(p))

const hasCategoryCountData = computed(() => {
  const catVars = Object.keys(summaries.value).filter(key => {
    if (isDerivedField(key)) return false
    const info = summaries.value[key]
    return info.type && ['categorical', 'categorical_numeric', 'ordinal'].includes(info.type)
  })
  return catVars.length > 0
})

const categoryCountOption = computed(() => {
  const catVars = Object.keys(summaries.value).filter(key => {
    if (isDerivedField(key)) return false
    const info = summaries.value[key]
    return info.type && ['categorical', 'categorical_numeric', 'ordinal'].includes(info.type)
  })
  if (catVars.length === 0) return {}
  const data = catVars.map(key => {
    const info = summaries.value[key]
    let vc = {}
    if (info.value_counts && Object.keys(info.value_counts).length > 0) {
      vc = info.value_counts
    } else if (info.top_categories && Object.keys(info.top_categories).length > 0) {
      vc = info.top_categories
    }
    return { name: key, value: Object.keys(vc).length }
  }).sort((a, b) => b.value - a.value)
  return {
    tooltip: { trigger: 'axis', formatter: (params) => `<strong>${params[0].name}</strong><br/>类别数：${params[0].value}` },
    grid: { left: '12%', right: '8%', top: '15%', bottom: '20%' },
    xAxis: { type: 'category', data: data.map(d => d.name), axisLabel: { rotate: 30, fontSize: 10, interval: 0 } },
    yAxis: { type: 'value', name: '类别数量' },
    series: [{ type: 'bar', data: data.map(d => ({ value: d.value, itemStyle: { color: '#9B59B6' } })), barWidth: '45%', label: { show: true, position: 'top', formatter: '{c}', fontSize: 10 } }]
  }
})

const hasContinuousRangeData = computed(() => {
  return Object.keys(summaries.value).filter(key => summaries.value[key]?.type === 'continuous').length > 0
})

const continuousRangeOption = computed(() => {
  const contVars = Object.keys(summaries.value).filter(key => summaries.value[key]?.type === 'continuous')
  if (contVars.length === 0) return {}
  const topVars = contVars.slice(0, 8)
  const colors = ['#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#9B59B6', '#1ABC9C', '#3498DB', '#2ECC71']
  const data = topVars.map((key, idx) => {
    const info = summaries.value[key]
    return {
      name: key,
      min: info.min !== undefined && info.min !== null ? Number(info.min) : 0,
      max: info.max !== undefined && info.max !== null ? Number(info.max) : 0,
      mean: info.mean !== undefined && info.mean !== null ? Number(info.mean) : 0,
      color: colors[idx % colors.length]
    }
  })
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const d = data.find(item => item.name === params[0].name)
        if (!d) return ''
        return `<strong>${d.name}</strong><br/>最小值：${d.min.toFixed(2)}<br/>最大值：${d.max.toFixed(2)}<br/>均值：${d.mean.toFixed(2)}`
      }
    },
    grid: { left: '12%', right: '8%', top: '10%', bottom: '20%' },
    xAxis: { type: 'category', data: data.map(d => d.name), axisLabel: { rotate: 30, fontSize: 10, interval: 0 } },
    yAxis: { type: 'value', name: '取值范围' },
    series: [{
      type: 'bar',
      name: '取值范围',
      data: data.map(d => ({
        value: d.max - d.min,
        itemStyle: { color: d.color, opacity: 0.3 }
      })),
      barWidth: '40%',
      label: {
        show: true,
        position: 'top',
        formatter: (params) => {
          const d = data[params.dataIndex]
          if (!d) return ''
          return `${d.min.toFixed(1)} ~ ${d.max.toFixed(1)}`
        },
        fontSize: 9
      }
    }]
  }
})

// ===== 工具函数 =====
function getRangeBarLeft(row) {
  const min = row._min !== undefined && row._min !== null ? Number(row._min) : 0
  const max = row._max !== undefined && row._max !== null ? Number(row._max) : 0
  const mean = row._mean !== undefined && row._mean !== null ? Number(row._mean) : 0
  const range = max - min
  if (range === 0) return '0%'
  return ((mean - min) / range * 100) + '%'
}

function getRangeBarWidth(row) {
  const min = row._min !== undefined && row._min !== null ? Number(row._min) : 0
  const max = row._max !== undefined && row._max !== null ? Number(row._max) : 0
  const range = max - min
  if (range === 0) return '100%'
  const width = (range / (Math.abs(max) + 1)) * 100
  return Math.max(width, 5) + '%'
}

function getRangeBarDot(row) {
  const min = row._min !== undefined && row._min !== null ? Number(row._min) : 0
  const max = row._max !== undefined && row._max !== null ? Number(row._max) : 0
  const mean = row._mean !== undefined && row._mean !== null ? Number(row._mean) : 0
  const range = max - min
  if (range === 0) return '50%'
  return ((mean - min) / range * 100) + '%'
}

function getCategoryColor(idx) {
  const colors = ['#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#9B59B6']
  return colors[idx % colors.length]
}

// ===== 字段详情弹窗 =====
function buildFieldData(fieldName) {
  const summary = summaries.value?.[fieldName] || {}
  const varType = variableTypes.value?.[fieldName]?.type || 'unknown'
  const varTypeDesc = variableTypes.value?.[fieldName]?.type_desc || typeDisplay[varType] || varType

  const tsDiag = currentData.value?.time_series_diagnostics?.[fieldName] || null
  const outlier = quality.value?.outliers?.[fieldName] || null
  const missing = quality.value?.missing?.find(m => m.column === fieldName) || null
  const dupInfo = quality.value?.duplicates || null

  const correlationsList = []
  const matrix = correlations.value?.matrix || {}
  if (matrix[fieldName]) {
    const entries = Object.entries(matrix[fieldName])
    for (const [varName, value] of entries) {
      if (varName !== fieldName && value !== null && value !== undefined && Math.abs(value) >= 0.7) {
        correlationsList.push({ var: varName, value: parseFloat(Number(value).toFixed(4)) })
      }
    }
    correlationsList.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  }

  const rules = []
  const auditRules = quality.value?.audit_rules || {}
  const allRules = [
    ...(auditRules.arithmetic_rules || []),
    ...(auditRules.functional_dependencies || []),
    ...(auditRules.temporal_rules || [])
  ]
  for (const rule of allRules) {
    if (rule.fields && rule.fields.includes(fieldName)) {
      rules.push({
        type: rule.relation_type === 'additive' ? '数值' :
              rule.rule?.includes('→') ? '函数依赖' : '时序约束',
        rule: rule.rule,
        confidence: rule.confidence || 1.0
      })
    }
  }

  const models = []
  const modelRecs = currentData.value?.model_recommendations || []
  for (const rec of modelRecs) {
    let role = ''
    if (rec.target_column === fieldName) {
      role = '🎯 目标'
    } else if (rec.feature_columns && rec.feature_columns.includes(fieldName)) {
      role = '📊 特征'
    }
    if (role) {
      models.push({
        role: role,
        task_type: rec.task_type || '',
        model: rec.ml || rec.model || '',
        target: rec.target_column || ''
      })
    }
  }

  let topCategories = []
  if (summary.top_categories && Object.keys(summary.top_categories).length > 0) {
    const entries = Object.entries(summary.top_categories)
    const total = entries.reduce((s, e) => s + e[1], 0)
    topCategories = entries.map(([name, count]) => ({
      name: String(name),
      count: count,
      pct: total > 0 ? (count / total * 100) : 0
    })).sort((a, b) => b.count - a.count)
  } else if (summary.value_counts && Object.keys(summary.value_counts).length > 0) {
    const entries = Object.entries(summary.value_counts)
    const total = entries.reduce((s, e) => s + e[1], 0)
    topCategories = entries.map(([name, count]) => ({
      name: String(name),
      count: count,
      pct: total > 0 ? (count / total * 100) : 0
    })).sort((a, b) => b.count - a.count)
  }

  return {
    fieldName,
    varType,
    varTypeDesc,
    summary: { ...summary, topCategories },
    tsDiag,
    outlier,
    missing,
    duplicateInfo: dupInfo,
    correlations: correlationsList,
    rules,
    models,
    topCategories
  }
}

function openFieldDetail(fieldName) {
  const data = buildFieldData(fieldName)
  fieldDetailStore.open(fieldName, data)
}

// ===== 数据预览联动 =====
function showMissingRows(fieldName) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }
  openDataPreview({
    sessionId: sessionId,
    title: `「${fieldName}」为空的数据`,
    filters: [
      { field: fieldName, condition: 'is_null', value: true }
    ]
  })
}

function showNonMissingRows(fieldName) {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.warning('请先加载项目')
    return
  }
  openDataPreview({
    sessionId: sessionId,
    title: `「${fieldName}」非空数据`,
    filters: [
      { field: fieldName, condition: 'is_not_null', value: true }
    ]
  })
}

// ===== 加载数据 =====
async function loadData() {
  let sessionId = sessionStore.currentSessionId
  if (!sessionId) {
    sessionId = localStorage.getItem('lastSessionId')
  }
  if (!sessionId) {
    loading.value = false
    return
  }

  loading.value = true
  try {
    const result = await reportApi.get(sessionId)
    reportData.value = result

    const allTables = result?.all_tables || {}
    const tableKeys = Object.keys(allTables)
    tableNames.value = tableKeys.filter(k => k !== 'merged')
    isMultiTable.value = tableNames.value.length > 1

    if (!currentTable.value || !allTables[currentTable.value]) {
      currentTable.value = 'merged'
    }
  } catch (err) {
    console.error('加载数据失败:', err)
    ElMessage.error('加载数据失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

function onTableChange() {
  catCountKey.value += 1
  contRangeKey.value += 1
}

function goToUpload() {
  router.push('/upload')
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.data-overview {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px 20px 80px 20px;
}
.subtitle {
  color: #909399;
  margin-bottom: 24px;
}
.loading-container {
  padding: 40px 0;
}
.empty-state {
  padding: 60px 0;
}

/* ===== 诊断卡片样式 ===== */
.diagnosis-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.diagnosis-card {
  border-left: 4px solid #409eff;
}
.diagnosis-card.danger {
  border-left-color: #f56c6c;
}
.diagnosis-card.warning {
  border-left-color: #e6a23c;
}
.diagnosis-card.info {
  border-left-color: #409eff;
}
.diagnosis-card .card-icon {
  font-size: 24px;
  flex-shrink: 0;
}
.diagnosis-card .card-content {
  flex: 1;
}
.diagnosis-card .card-title {
  font-weight: 600;
  font-size: 14px;
  color: #2c3e50;
}
.diagnosis-card .card-desc {
  font-size: 12px;
  color: #909399;
  margin-top: 2px;
}
.diagnosis-card .card-tags {
  margin-top: 6px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

/* ===== 原有样式 ===== */
.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
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

.charts-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
}
.chart-card {
  background: #fff;
  border-radius: 12px;
  border: 1px solid #e4e7ed;
  padding: 16px 16px 4px 16px;
  transition: box-shadow 0.2s;
}
.chart-card:hover {
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.chart-card.full-width {
  grid-column: 1 / -1;
  margin-bottom: 16px;
}
.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.chart-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
}
.chart-container {
  width: 100%;
  height: 220px;
}
.chart-empty {
  height: 180px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #bbb;
  font-size: 13px;
}
@media (max-width: 768px) {
  .charts-row {
    grid-template-columns: 1fr;
  }
  .chart-container {
    height: 200px;
  }
}

.section {
  margin-top: 30px;
}
.section h4 {
  margin-bottom: 12px;
  color: #2c3e50;
  font-size: 16px;
}
.field-tabs {
  margin-top: 4px;
}
.empty-tip {
  padding: 20px;
  text-align: center;
  color: #909399;
}

.field-name-link {
  color: #409EFF;
  cursor: pointer;
  font-weight: 500;
  transition: color 0.2s;
}
.field-name-link:hover {
  color: #66b1ff;
  text-decoration: underline;
}

.range-bar-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding: 2px 0;
}
.range-bar-track {
  position: relative;
  width: 100%;
  height: 6px;
  background: #e8ecf1;
  border-radius: 3px;
}
.range-bar-fill {
  position: absolute;
  top: 0;
  height: 100%;
  background: rgba(64, 158, 255, 0.25);
  border-radius: 3px;
  min-width: 2px;
}
.range-bar-dot {
  position: absolute;
  top: -4px;
  width: 14px;
  height: 14px;
  background: #409EFF;
  border-radius: 50%;
  border: 2px solid white;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
  transform: translateX(-7px);
  cursor: pointer;
}
.range-bar-labels {
  display: flex;
  justify-content: space-between;
  width: 100%;
  font-size: 9px;
  color: #909399;
  margin-top: 2px;
}
.range-min, .range-max {
  font-size: 9px;
  color: #909399;
}

.category-bars {
  display: flex;
  flex-direction: column;
  gap: 3px;
  width: 100%;
}
.category-bar-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
}
.category-name {
  min-width: 20px;
  max-width: 40px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #555;
  font-size: 10px;
}
.category-bar-track {
  flex: 1;
  height: 4px;
  background: #e8ecf1;
  border-radius: 2px;
  overflow: hidden;
}
.category-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.3s;
  min-width: 2px;
}
.category-pct {
  font-size: 10px;
  color: #909399;
  min-width: 32px;
  text-align: right;
}
.text-muted {
  color: #bbb;
  font-size: 12px;
}
</style>