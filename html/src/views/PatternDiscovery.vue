<template>
  <div class="pattern-discovery">
    <h2>📈 规律发现</h2>
    <p class="subtitle">探索变量之间的相关性和时间序列规律</p>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="reportData" class="pattern-content">
      <el-tabs v-model="activeTab">
        <!-- 相关性分析 -->
        <el-tab-pane label="相关性分析" name="correlation">
          <div v-if="highCorrelations.length === 0" class="empty-tip">
            未发现强相关关系（|r| > 0.7）
          </div>
          <div v-else>
            <p>发现 <strong>{{ highCorrelations.length }}</strong> 对强相关关系：</p>
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

            <!-- 相关性热力图占位 -->
            <div class="plot-placeholder">
              <el-alert
                title="相关性热力图"
                type="info"
                :closable="false"
                show-icon
                description="后端返回 base64 图片时在此显示"
              />
            </div>
          </div>
        </el-tab-pane>

        <!-- 时间序列分析 -->
        <el-tab-pane label="时间序列分析" name="timeseries">
          <div v-if="timeSeriesData.length === 0" class="empty-tip">
            未检测到时间序列数据
          </div>
          <div v-else>
            <el-table :data="timeSeriesData" border size="small">
              <el-table-column prop="key" label="变量/分组" />
              <el-table-column prop="n_samples" label="样本量" width="100" />
              <el-table-column prop="stationary" label="平稳性" width="120" />
              <el-table-column prop="autocorrelation" label="自相关性" width="120" />
              <el-table-column prop="seasonality" label="季节性" width="100" />
            </el-table>
            <div class="plot-placeholder">
              <el-alert
                title="时间序列图表"
                type="info"
                :closable="false"
                show-icon
                description="后端返回 base64 图片时在此显示"
              />
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>

    <div v-else class="empty-state">
      <el-empty description="请先完成数据分析">
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
const activeTab = ref('correlation')

const highCorrelations = computed(() => {
  return reportData.value?.correlations?.high_correlations || []
})

const timeSeriesData = computed(() => {
  const diag = reportData.value?.time_series_diagnostics || {}
  return Object.entries(diag).map(([key, info]) => ({
    key,
    n_samples: info.n_samples || 0,
    stationary: info.is_stationary ? '✅ 平稳' : '⚠️ 非平稳',
    autocorrelation: info.has_autocorrelation ? '✅ 有' : '❌ 无',
    seasonality: info.has_seasonality ? '✅ 有' : '❌ 无'
  }))
})

async function loadData() {
  let sessionId = sessionStore.currentSessionId
  if (!sessionId) {
    sessionId = localStorage.getItem('lastSessionId')
  }
  if (!sessionId) {
    router.push('/')
    return
  }
  if (!sessionStore.currentSessionId) {
    sessionStore.currentSessionId = sessionId
  }

  loading.value = true
  try {
    const result = await reportApi.get(sessionId)
    reportData.value = result
  } catch (err) {
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
.pattern-discovery {
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
.empty-tip {
  padding: 40px;
  text-align: center;
  color: #909399;
  font-size: 16px;
}
.plot-placeholder {
  margin-top: 20px;
}
</style>