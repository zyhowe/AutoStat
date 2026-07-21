<template>
  <div class="scenario-derivation">
    <div class="page-header">
      <h2>🧩 场景推导</h2>
      <p class="subtitle">基于技术分析结果，审核并确认要执行的业务场景</p>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="8" animated />
    </div>

    <!-- 错误状态 -->
    <div v-else-if="error" class="error-container">
      <el-result icon="error" :title="error" sub-title="请先完成数据分析">
        <template #extra>
          <el-button type="primary" @click="goToUpload">去上传数据</el-button>
        </template>
      </el-result>
    </div>

    <!-- 场景列表 -->
    <div v-else class="scenario-container">
      <!-- 顶部操作栏 -->
      <div class="toolbar">
        <div class="toolbar-left">
          <el-tag :type="status === 'executed' ? 'success' : 'info'">
            {{ status === 'executed' ? '✅ 已执行' : '📝 待确认' }}
          </el-tag>
          <span class="scenario-count">共 {{ candidates.length }} 个候选场景</span>
        </div>
        <div class="toolbar-right">
          <el-button size="small" @click="loadData">🔄 刷新</el-button>
          <el-button
            size="small"
            type="primary"
            :loading="executing"
            :disabled="candidates.length === 0 || status === 'executed'"
            @click="handleExecuteAll"
          >
            {{ executing ? '执行中...' : '🚀 确认执行' }}
          </el-button>
        </div>
      </div>

      <!-- 场景卡片列表 -->
      <div class="scenario-grid">
        <el-card
          v-for="(scenario, index) in candidates"
          :key="scenario.id"
          class="scenario-card"
          shadow="hover"
          :class="{ disabled: !scenario.enabled, executed: status === 'executed' }"
        >
          <!-- 卡片头部 -->
          <div class="card-header">
            <div class="header-left">
              <span class="category-tag">{{ scenario.category || '通用' }}</span>
              <span class="scenario-id">{{ scenario.id }}</span>
              <span class="scenario-name">{{ scenario.name }}</span>
            </div>
            <div class="header-right">
              <el-switch
                v-model="scenario.enabled"
                :disabled="status === 'executed'"
                size="small"
                @change="onScenarioChange"
              />
              <el-button
                size="small"
                text
                type="danger"
                :disabled="status === 'executed'"
                @click="removeScenario(index)"
              >
                🗑️
              </el-button>
            </div>
          </div>

          <!-- 卡片内容 -->
          <div class="card-body">
            <div class="trigger-basis">
              <span class="label">📌 触发依据：</span>
              <span class="value">{{ scenario.trigger_basis || '技术特征满足触发条件' }}</span>
            </div>
            <div class="scenario-desc">
              <span class="label">📋 说明：</span>
              <span class="value">{{ scenario.description || '自动推导的业务场景' }}</span>
            </div>
            <div class="scenario-params" v-if="Object.keys(scenario.params || {}).length > 0">
              <span class="label">⚙️ 参数：</span>
              <el-tag
                v-for="(value, key) in scenario.params"
                :key="key"
                size="small"
                type="info"
                style="margin: 2px;"
              >
                {{ key }}={{ value }}
              </el-tag>
            </div>
          </div>

          <!-- 卡片底部 -->
          <div class="card-footer" v-if="status === 'executed' && scenarioResults.length > 0">
            <el-tag type="success" size="small">✅ 已执行</el-tag>
            <span class="result-preview" v-if="getScenarioResult(scenario.id)">
              {{ getScenarioResult(scenario.id) }}
            </span>
          </div>
        </el-card>
      </div>

      <!-- 空状态 -->
      <div v-if="candidates.length === 0 && !loading" class="empty-state">
        <el-empty description="暂未推导出场景，请检查数据是否完成分析">
          <el-button type="primary" @click="goToUpload">去上传数据</el-button>
        </el-empty>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { scenariosApi } from '../api/scenarios'

const router = useRouter()
const sessionStore = useSessionStore()

const loading = ref(false)
const executing = ref(false)
const error = ref('')
const candidates = ref([])
const results = ref([])
const status = ref('draft')

const scenarioResults = computed(() => results.value || [])

onMounted(() => {
  loadData()
})

async function loadData() {
  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    error.value = '请先加载项目'
    return
  }

  loading.value = true
  error.value = ''
  try {
    const response = await scenariosApi.get(sessionId)
    candidates.value = response.candidates || []
    results.value = response.results || []
    status.value = response.status || 'draft'
  } catch (err) {
    error.value = err.message || '加载失败'
  } finally {
    loading.value = false
  }
}

function onScenarioChange() {
  // 标记已修改，但不自动保存
}

async function removeScenario(index) {
  try {
    await ElMessageBox.confirm('确定要删除这个场景吗？', '确认删除', {
      type: 'warning'
    })
    candidates.value.splice(index, 1)
    ElMessage.success('已删除')
  } catch (err) {
    // 取消
  }
}

async function handleExecuteAll() {
  const enabledCount = candidates.value.filter(s => s.enabled).length
  if (enabledCount === 0) {
    ElMessage.warning('没有启用任何场景')
    return
  }

  try {
    await ElMessageBox.confirm(
      `将执行 ${enabledCount} 个已启用的场景，确认继续？`,
      '确认执行',
      { type: 'info' }
    )
  } catch {
    return
  }

  const sessionId = sessionStore.currentSessionId || localStorage.getItem('lastSessionId')
  if (!sessionId) {
    ElMessage.error('请先加载项目')
    return
  }

  executing.value = true
  try {
    // 先保存场景配置
    await scenariosApi.update(sessionId, candidates.value)

    // 执行场景
    const response = await scenariosApi.execute(sessionId)
    results.value = response.results || []
    status.value = 'executed'
    ElMessage.success(`成功执行 ${response.count || 0} 个场景`)

    // 跳转到仪表板
    router.push('/scenario-dashboard')
  } catch (err) {
    ElMessage.error(err.message || '执行失败')
  } finally {
    executing.value = false
  }
}

function getScenarioResult(scenarioId) {
  const result = results.value.find(r => r.scenario_id === scenarioId)
  if (result && result.conclusions && result.conclusions.length > 0) {
    const summary = result.conclusions.find(c => c.type === 'summary')
    return summary ? summary.text : result.conclusions[0]?.text
  }
  return null
}

function goToUpload() {
  router.push('/upload')
}
</script>

<style scoped>
.scenario-derivation {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.page-header {
  margin-bottom: 24px;
}
.page-header h2 {
  margin: 0 0 8px 0;
  color: #2c3e50;
}
.subtitle {
  color: #909399;
  margin: 0;
}

.loading-container {
  padding: 40px 0;
}
.error-container {
  padding: 60px 0;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 12px 16px;
  background: #f5f7fa;
  border-radius: 8px;
}
.toolbar-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.scenario-count {
  font-size: 13px;
  color: #909399;
}
.toolbar-right {
  display: flex;
  gap: 8px;
}

.scenario-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.scenario-card {
  transition: all 0.2s;
}
.scenario-card:hover {
  transform: translateY(-2px);
}
.scenario-card.disabled {
  opacity: 0.5;
}
.scenario-card.executed {
  border-left: 4px solid #67c23a;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.category-tag {
  font-size: 11px;
  color: #fff;
  background: #409eff;
  padding: 2px 8px;
  border-radius: 4px;
}
.scenario-id {
  font-size: 11px;
  font-weight: 600;
  color: #909399;
  background: #f0f2f6;
  padding: 2px 8px;
  border-radius: 4px;
}
.scenario-name {
  font-size: 15px;
  font-weight: 600;
  color: #2c3e50;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 4px;
}

.card-body {
  font-size: 13px;
  color: #555;
  line-height: 1.6;
}
.card-body .label {
  color: #909399;
}
.card-body .value {
  color: #2c3e50;
}
.trigger-basis, .scenario-desc, .scenario-params {
  padding: 4px 0;
}
.scenario-params {
  margin-top: 4px;
}

.card-footer {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
  color: #67c23a;
}
.result-preview {
  color: #555;
  font-size: 12px;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.empty-state {
  padding: 60px 0;
}

@media (max-width: 768px) {
  .scenario-grid {
    grid-template-columns: 1fr;
  }
  .toolbar {
    flex-direction: column;
    gap: 12px;
    align-items: stretch;
  }
  .toolbar-right {
    justify-content: flex-end;
  }
}
</style>