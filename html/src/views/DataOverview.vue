<template>
  <div class="data-overview">
    <h2>📊 数据概览</h2>
    <p class="subtitle">查看数据的基本信息和字段详情</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="overview-content">
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-value">{{ reportData.data_shape?.rows || 0 }}</div>
          <div class="stat-label">总行数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ reportData.data_shape?.columns || 0 }}</div>
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

      <!-- 变量类型分布 -->
      <div class="section">
        <h4>📋 变量类型分布</h4>
        <div class="type-tags">
          <el-tag
            v-for="(count, type) in typeCounts"
            :key="type"
            size="large"
            class="type-tag"
          >
            {{ typeDisplay[type] || type }}：{{ count }}
          </el-tag>
        </div>
      </div>

      <!-- 变量详情表格 -->
      <div class="section">
        <h4>📋 字段详情</h4>
        <el-table :data="variableList" border size="small" max-height="420" style="width: 100%;">
          <el-table-column prop="name" label="字段名" width="140" fixed="left" show-overflow-tooltip />
          <el-table-column prop="type_desc" label="类型" width="100" align="center" />
          <el-table-column prop="count" label="样本量" width="90" align="center" />
          <el-table-column prop="missing" label="缺失数" width="90" align="center" />
          <el-table-column prop="missing_pct" label="缺失率" width="90" align="center">
            <template #default="{ row }">
              {{ row.missing_pct.toFixed(1) }}%
            </template>
          </el-table-column>
          <el-table-column prop="center" label="中心趋势" width="120" align="center" show-overflow-tooltip />
          <el-table-column prop="spread" label="分布" min-width="120" show-overflow-tooltip />
        </el-table>
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
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { reportApi } from '../api/report'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(false)
const reportData = ref(null)

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
  const variableTypes = reportData.value?.variable_types || {}
  const counts = {}
  for (const info of Object.values(variableTypes)) {
    const typ = info.type || 'unknown'
    counts[typ] = (counts[typ] || 0) + 1
  }
  return counts
})

const variableTypesCount = computed(() => Object.keys(typeCounts.value).length)

const missingFieldsCount = computed(() => {
  return reportData.value?.quality_report?.missing?.length || 0
})

const variableList = computed(() => {
  const summaries = reportData.value?.variable_summaries || {}
  return Object.entries(summaries).map(([name, info]) => ({
    name,
    type_desc: info.type_desc || info.type,
    count: info.count || 0,
    missing: info.missing || 0,
    missing_pct: info.missing_pct || 0,
    center: info.mean !== undefined ? info.mean.toFixed(2) : (info.mode || '-'),
    spread: info.std !== undefined ? `±${info.std.toFixed(2)}` : (info.n_unique ? `${info.n_unique}个类别` : '-')
  })).slice(0, 30)
})

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
  } catch (err) {
    console.error('加载数据失败:', err)
    ElMessage.error('加载数据失败: ' + err.message)
  } finally {
    loading.value = false
  }
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
  padding: 20px;
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

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 30px;
}
.stat-card {
  background: #f5f7fa;
  border-radius: 12px;
  padding: 20px;
  text-align: center;
}
.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: #2c3e50;
}
.stat-label {
  font-size: 13px;
  color: #909399;
  margin-top: 4px;
}

.section {
  margin-bottom: 30px;
}
.section h4 {
  margin-bottom: 12px;
  color: #2c3e50;
  font-size: 16px;
}
.type-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.type-tag {
  font-size: 14px;
  padding: 8px 16px;
}
</style>