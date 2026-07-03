<template>
  <div class="details-tab">
    <h3>📋 详细分析</h3>

    <el-collapse v-model="activeNames">
      <!-- 数据概览 -->
      <el-collapse-item name="overview">
        <template #title>
          <span>📊 数据概览</span>
        </template>
        <div class="detail-content">
          <el-descriptions :column="4" border>
            <el-descriptions-item label="总行数">{{ reportData?.data_shape?.rows || 0 }}</el-descriptions-item>
            <el-descriptions-item label="总列数">{{ reportData?.data_shape?.columns || 0 }}</el-descriptions-item>
            <el-descriptions-item label="缺失字段">{{ reportData?.quality_report?.missing?.length || 0 }}</el-descriptions-item>
            <el-descriptions-item label="重复记录">{{ reportData?.quality_report?.duplicates?.count || 0 }}</el-descriptions-item>
          </el-descriptions>
          <div class="type-distribution">
            <span class="label">变量类型分布：</span>
            <span v-for="(count, type) in typeCounts" :key="type" class="type-tag">
              {{ typeDisplay[type] || type }}: {{ count }}
            </span>
          </div>
        </div>
      </el-collapse-item>

      <!-- 变量详情 -->
      <el-collapse-item name="variables">
        <template #title>
          <span>📋 变量详情</span>
        </template>
        <div class="detail-content">
          <el-table :data="variableList" border size="small" max-height="300">
            <el-table-column prop="name" label="变量名" />
            <el-table-column prop="type_desc" label="类型" />
            <el-table-column prop="count" label="样本量" />
            <el-table-column prop="missing" label="缺失数" />
            <el-table-column prop="missing_pct" label="缺失率">
              <template #default="{ row }">
                {{ row.missing_pct.toFixed(1) }}%
              </template>
            </el-table-column>
            <el-table-column prop="center" label="中心趋势" />
            <el-table-column prop="spread" label="分布" />
          </el-table>
        </div>
      </el-collapse-item>

      <!-- 相关性分析 -->
      <el-collapse-item name="correlation">
        <template #title>
          <span>🔗 相关性分析</span>
        </template>
        <div class="detail-content">
          <div v-if="highCorrelations.length > 0">
            <p>发现 <strong>{{ highCorrelations.length }}</strong> 对强相关关系（|r| > 0.7）：</p>
            <el-table :data="highCorrelations" border size="small">
              <el-table-column prop="var1" label="变量1" />
              <el-table-column prop="var2" label="变量2" />
              <el-table-column prop="value" label="相关系数" width="120">
                <template #default="{ row }">
                  {{ row.value.toFixed(3) }}
                </template>
              </el-table-column>
              <el-table-column label="方向" width="80">
                <template #default="{ row }">
                  <el-tag :type="row.value > 0 ? 'danger' : 'success'" size="small">
                    {{ row.value > 0 ? '正相关' : '负相关' }}
                  </el-tag>
                </template>
              </el-table-column>
            </el-table>
          </div>
          <div v-else>
            <p>未发现强相关关系（|r| > 0.7）</p>
          </div>
        </div>
      </el-collapse-item>

      <!-- 时间序列分析 -->
      <el-collapse-item name="timeseries">
        <template #title>
          <span>📈 时间序列分析</span>
        </template>
        <div class="detail-content">
          <div v-if="timeSeriesData.length > 0">
            <el-table :data="timeSeriesData" border size="small">
              <el-table-column prop="key" label="变量/分组" />
              <el-table-column prop="n_samples" label="样本量" />
              <el-table-column prop="stationary" label="平稳性" />
              <el-table-column prop="autocorrelation" label="自相关性" />
              <el-table-column prop="seasonality" label="季节性" />
            </el-table>
          </div>
          <div v-else>
            <p>未检测到时间序列数据</p>
          </div>
        </div>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  reportData: { type: Object, default: () => ({}) },
  qualityData: { type: Object, default: () => ({}) }
})

const activeNames = ref(['overview'])

const typeDisplay = {
  continuous: '连续变量',
  categorical: '分类变量',
  categorical_numeric: '数值型分类',
  ordinal: '有序分类',
  datetime: '日期时间',
  identifier: '标识符',
  text: '文本'
}

const typeCounts = computed(() => {
  const variableTypes = props.reportData?.variable_types || {}
  const counts = {}
  for (const info of Object.values(variableTypes)) {
    const typ = info.type || 'unknown'
    counts[typ] = (counts[typ] || 0) + 1
  }
  return counts
})

const variableList = computed(() => {
  const summaries = props.reportData?.variable_summaries || {}
  return Object.entries(summaries).map(([name, info]) => ({
    name,
    type_desc: info.type_desc || info.type,
    count: info.count || 0,
    missing: info.missing || 0,
    missing_pct: info.missing_pct || 0,
    center: info.mean !== undefined ? info.mean.toFixed(2) : (info.mode || '-'),
    spread: info.std !== undefined ? `±${info.std.toFixed(2)}` : (info.n_unique ? `${info.n_unique}个类别` : '-')
  })).slice(0, 20)
})

const highCorrelations = computed(() => {
  return props.reportData?.correlations?.high_correlations || []
})

const timeSeriesData = computed(() => {
  const diag = props.reportData?.time_series_diagnostics || {}
  return Object.entries(diag).map(([key, info]) => ({
    key,
    n_samples: info.n_samples || 0,
    stationary: info.is_stationary ? '✅ 平稳' : '⚠️ 非平稳',
    autocorrelation: info.has_autocorrelation ? '✅ 有' : '❌ 无',
    seasonality: info.has_seasonality ? '✅ 有' : '❌ 无'
  }))
})
</script>

<style scoped>
.details-tab h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}
.detail-content {
  padding: 10px 0;
}
.type-distribution {
  margin-top: 12px;
  font-size: 13px;
}
.type-distribution .label {
  color: #666;
}
.type-distribution .type-tag {
  display: inline-block;
  margin: 0 8px 4px 0;
  background: #f0f2f6;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 12px;
}
</style>