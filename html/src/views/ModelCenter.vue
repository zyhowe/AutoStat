<template>
  <div class="model-center">
    <h2>🤖 智能预测</h2>
    <p class="subtitle">模型推荐、训练、预测一体化管理</p>

    <el-tabs v-model="activeTab">
      <!-- ==================== 训练 ==================== -->
      <el-tab-pane label="🏋️ 训练" name="train">
        <div class="train-section">
          <el-alert
            v-if="!sessionStore.hasSession"
            title="请先上传数据并完成分析"
            type="warning"
            show-icon
            :closable="false"
          />
          <div v-else>
            <!-- 模型推荐列表 -->
            <div class="recommendations">
              <h4>📊 推荐模型 <el-tag size="small" type="info">点击卡片打开训练配置</el-tag></h4>
              <div v-if="recommendations.length === 0" class="empty-hint">
                暂无模型推荐，请先完成数据分析
              </div>
              <div v-else class="recommend-list">
                <el-card
                  v-for="(rec, idx) in recommendations"
                  :key="idx"
                  class="recommend-card"
                  shadow="hover"
                  @click="openTrainDialog(rec, idx)"
                >
                  <div class="rec-priority" :class="rec.priority">
                    {{ rec.priority === '高' ? '🔴' : rec.priority === '中' ? '🟠' : '🟢' }}
                  </div>
                  <div class="rec-task">{{ rec.task_type }}</div>
                  <div class="rec-model">{{ rec.ml }}</div>
                  <div class="rec-target" v-if="rec.target_column">🎯 {{ rec.target_column }}</div>
                  <div class="rec-features" v-if="rec.feature_columns && rec.feature_columns.length > 0">
                    📊 {{ rec.feature_columns.slice(0, 4).join('、') }}{{ rec.feature_columns.length > 4 ? `等${rec.feature_columns.length}个` : '' }}
                  </div>
                  <div class="rec-reason" v-if="rec.reason">💡 {{ rec.reason }}</div>
                  <el-button size="small" type="primary" class="rec-btn">📋 配置训练</el-button>
                </el-card>
              </div>
            </div>
          </div>
        </div>
      </el-tab-pane>

      <!-- ==================== 预测 ==================== -->
      <el-tab-pane label="🔮 预测" name="predict">
        <div class="predict-section">
          <div v-if="savedModels.length === 0" class="empty-state">
            <el-empty description="暂无已训练的模型，请先训练模型">
              <el-button type="primary" @click="activeTab = 'train'">去训练</el-button>
            </el-empty>
          </div>
          <div v-else>
            <el-form label-width="120px">
              <el-form-item label="选择模型">
                <el-select v-model="predictForm.modelKey" @change="onModelSelect" style="width: 100%;">
                  <el-option
                    v-for="model in savedModels"
                    :key="model.model_key"
                    :label="model.user_model_name || model.model_key"
                    :value="model.model_key"
                  />
                </el-select>
              </el-form-item>

              <div v-if="selectedModel" class="model-info">
                <el-descriptions :column="2" border size="small">
                  <el-descriptions-item label="类型">{{ selectedModel.task_type }}</el-descriptions-item>
                  <el-descriptions-item label="目标">{{ selectedModel.target_column || '无' }}</el-descriptions-item>
                  <el-descriptions-item label="特征" :span="2">
                    {{ (selectedModel.features || []).join('、') }}
                  </el-descriptions-item>
                  <el-descriptions-item label="指标" :span="2" v-if="selectedModel.metrics">
                    {{ formatMetrics(selectedModel.metrics) }}
                  </el-descriptions-item>
                </el-descriptions>
              </div>

              <div v-if="selectedModel && selectedModel.features" class="input-fields">
                <el-form-item
                  v-for="feature in selectedModel.features"
                  :key="feature"
                  :label="feature"
                >
                  <el-input
                    v-model="predictForm.inputValues[feature]"
                    placeholder="输入值"
                  />
                </el-form-item>
              </div>
            </el-form>

            <el-button type="primary" :loading="predicting" :disabled="!canPredict" @click="handlePredict">
              🔍 执行预测
            </el-button>

            <!-- 预测结果 -->
            <div v-if="predictResult" class="predict-result">
              <el-divider />
              <h4>📊 预测结果</h4>
              <el-descriptions :column="2" border>
                <el-descriptions-item label="预测值">
                  <span class="result-value">{{ predictResult.prediction }}</span>
                </el-descriptions-item>
                <el-descriptions-item label="置信度" v-if="predictResult.confidence !== undefined && predictResult.confidence !== null">
                  {{ (predictResult.confidence * 100).toFixed(2) }}%
                </el-descriptions-item>
                <el-descriptions-item label="模型" :span="2">
                  {{ predictResult.model_name }}
                </el-descriptions-item>
                <el-descriptions-item label="概率分布" :span="2" v-if="predictResult.probabilities && predictResult.probabilities.length > 0">
                  <div class="prob-dist">
                    <span v-for="(p, idx) in predictResult.probabilities" :key="idx">
                      类别 {{ idx }}: {{ (p * 100).toFixed(1) }}%
                    </span>
                  </div>
                </el-descriptions-item>
              </el-descriptions>
            </div>
          </div>
        </div>
      </el-tab-pane>
    </el-tabs>

    <!-- ==================== 训练配置弹窗 ==================== -->
    <el-dialog
      v-model="trainDialogVisible"
      :title="`⚙️ 训练配置 - ${selectedRec?.task_type || ''}`"
      width="780px"
      destroy-on-close
      :close-on-click-modal="false"
      :close-on-press-escape="!training"
    >
      <div v-if="selectedRec" class="dialog-body">
        <!-- 推荐摘要 -->
        <div class="dialog-summary">
          <el-alert
            :title="`${selectedRec.task_type}：${selectedRec.ml}`"
            :description="selectedRec.reason || '基于数据特征推荐'"
            type="info"
            show-icon
            :closable="false"
          />
        </div>

        <!-- 训练表单 -->
        <el-form label-width="120px" :disabled="training">
          <el-form-item label="任务类型">
            <el-select v-model="trainForm.taskType" @change="onTaskTypeChange" style="width: 100%;">
              <el-option label="📊 分类" value="classification" />
              <el-option label="📈 回归" value="regression" />
              <el-option label="🔘 聚类" value="clustering" />
              <el-option label="📅 时间序列" value="time_series" />
            </el-select>
          </el-form-item>

          <el-form-item label="目标列" v-if="trainForm.taskType !== 'clustering'">
            <el-select v-model="trainForm.targetColumn" placeholder="选择目标列" style="width: 100%;">
              <el-option
                v-for="col in numericColumns"
                :key="col"
                :label="col"
                :value="col"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="特征列">
            <el-select
              v-model="trainForm.features"
              multiple
              filterable
              placeholder="选择特征列"
              style="width: 100%;"
            >
              <el-option
                v-for="col in allColumns"
                :key="col"
                :label="col"
                :value="col"
                :disabled="col === trainForm.targetColumn"
              />
            </el-select>
            <span class="hint">已选 {{ trainForm.features.length }} 个特征</span>
          </el-form-item>

          <el-form-item label="模型">
            <el-select v-model="trainForm.modelKey" @change="onModelChange" style="width: 100%;">
              <el-option
                v-for="model in availableModels"
                :key="model.key"
                :label="model.name"
                :value="model.key"
              />
            </el-select>
            <span class="hint">{{ getModelDescription() }}</span>
          </el-form-item>

          <el-form-item label="模型参数" v-if="modelParams.length > 0">
            <div class="params-grid">
              <div
                v-for="param in modelParams"
                :key="param.name"
                class="param-item"
              >
                <label>{{ param.label }}</label>
                <el-input-number
                  v-if="param.type === 'number'"
                  v-model="trainForm.params[param.name]"
                  :min="param.min || 0"
                  :max="param.max || 9999"
                  :step="param.step || 1"
                  size="small"
                  style="width: 100%;"
                />
                <el-input
                  v-else-if="param.type === 'text'"
                  v-model="trainForm.params[param.name]"
                  size="small"
                  :placeholder="param.placeholder || ''"
                />
                <el-select
                  v-else-if="param.type === 'select'"
                  v-model="trainForm.params[param.name]"
                  size="small"
                  style="width: 100%;"
                >
                  <el-option
                    v-for="opt in param.options"
                    :key="opt"
                    :label="opt"
                    :value="opt"
                  />
                </el-select>
                <el-switch
                  v-else-if="param.type === 'boolean'"
                  v-model="trainForm.params[param.name]"
                  size="small"
                />
                <span class="param-hint">{{ param.hint }}</span>
              </div>
            </div>
          </el-form-item>

          <el-form-item label="模型名称">
            <el-input v-model="trainForm.userModelName" placeholder="自动生成" />
          </el-form-item>
        </el-form>

        <!-- 训练进度和日志 -->
        <div v-if="training || trainLogs.length > 0" class="train-progress-dialog">
          <el-divider />
          <el-progress :percentage="trainProgress" :format="formatProgress" />
          <p class="status-message">{{ trainStatusMessage }}</p>

          <div v-if="trainLogs.length > 0" class="train-logs">
            <h4>📋 训练日志</h4>
            <div class="log-container">
              <div
                v-for="(log, idx) in trainLogs"
                :key="idx"
                class="log-line"
                :class="log.type"
              >
                <span class="log-time">{{ log.time }}</span>
                <span class="log-content">{{ log.message }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <template #footer>
        <div class="dialog-footer">
          <el-button @click="closeDialog" :disabled="training">取消</el-button>
          <el-button
            type="primary"
            :loading="training"
            :disabled="!canTrain || training"
            @click="handleTrain"
          >
            {{ training ? '训练中...' : '🚀 开始训练' }}
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useSessionStore } from '../stores/session'
import { modelsApi } from '../api/models'
import { reportApi } from '../api/report'

const sessionStore = useSessionStore()
const reportData = ref(null)

// ==================== 状态 ====================
const activeTab = ref('train')
const training = ref(false)
const trainProgress = ref(0)
const trainStatusMessage = ref('')
const trainLogs = ref([])
const predicting = ref(false)
const predictResult = ref(null)

const recommendations = ref([])
const selectedRec = ref(null)
const trainDialogVisible = ref(false)

const trainForm = ref({
  taskType: 'classification',
  targetColumn: '',
  features: [],
  modelKey: '',
  userModelName: '',
  params: {}
})

const predictForm = ref({
  modelKey: '',
  inputValues: {}
})

const savedModels = ref([])
const selectedModel = ref(null)

const availableModels = ref([])
const modelParams = ref([])

const allColumns = ref([])
const numericColumns = ref([])

// ==================== 计算属性 ====================
const canTrain = computed(() => {
  return trainForm.value.features.length > 0 &&
    (trainForm.value.taskType === 'clustering' || trainForm.value.targetColumn) &&
    trainForm.value.modelKey
})

const canPredict = computed(() => {
  return predictForm.value.modelKey &&
    Object.keys(predictForm.value.inputValues).length > 0 &&
    Object.values(predictForm.value.inputValues).some(v => v !== '' && v !== undefined && v !== null)
})

// ==================== 生命周期 ====================
onMounted(async () => {
  await loadColumns()
  await loadModels()
  await loadRecommendations()
})

// ==================== 加载数据 ====================
async function loadColumns() {
  const session = sessionStore.currentSession
  if (session && session.variable_types) {
    allColumns.value = Object.keys(session.variable_types)
    numericColumns.value = Object.keys(session.variable_types).filter(
      col => session.variable_types[col] === 'continuous'
    )
  }
}

async function loadModels() {
  if (!sessionStore.currentSessionId) return
  try {
    const result = await modelsApi.list(sessionStore.currentSessionId)
    savedModels.value = result || []
  } catch (err) {
    console.error('加载模型失败:', err)
  }
}

async function loadRecommendations() {
  if (!sessionStore.currentSessionId) return
  try {
    const sessionId = sessionStore.currentSessionId
    const result = await reportApi.get(sessionId)
    if (result && result.model_recommendations) {
      recommendations.value = result.model_recommendations.slice(0, 10)
    } else {
      recommendations.value = []
    }
  } catch (err) {
    console.error('加载推荐失败:', err)
    recommendations.value = []
  }
}

// ==================== 打开/关闭训练弹窗 ====================
function openTrainDialog(rec, index) {
  selectedRec.value = rec

  const taskTypeMap = {
    '回归预测': 'regression',
    '分类预测': 'classification',
    '时间序列预测': 'time_series',
    '聚类分析': 'clustering'
  }

  let taskType = 'classification'
  for (const [key, value] of Object.entries(taskTypeMap)) {
    if (rec.task_type && rec.task_type.includes(key)) {
      taskType = value
      break
    }
  }

  trainForm.value.taskType = taskType
  trainForm.value.targetColumn = rec.target_column || ''
  trainForm.value.features = rec.feature_columns || []

  const modelMap = {
    'classification': 'random_forest',
    'regression': 'random_forest_regressor',
    'clustering': 'kmeans',
    'time_series': 'arima'
  }
  trainForm.value.modelKey = modelMap[taskType] || 'random_forest'

  const targetName = rec.target_column || 'model'
  const modelName = rec.ml?.split('/')[0]?.trim() || 'model'
  trainForm.value.userModelName = `${taskType}_${targetName}_${modelName}`

  trainLogs.value = []
  trainProgress.value = 0
  trainStatusMessage.value = ''

  onTaskTypeChange()
  onModelChange()

  trainDialogVisible.value = true
}

function closeDialog() {
  if (training.value) {
    ElMessage.warning('训练正在进行中，请等待完成')
    return
  }
  trainDialogVisible.value = false
}

// ==================== 任务类型切换 ====================
function onTaskTypeChange() {
  const modelMap = {
    classification: [
      { key: 'logistic_regression', name: '逻辑回归' },
      { key: 'decision_tree', name: '决策树' },
      { key: 'random_forest', name: '随机森林' },
      { key: 'xgboost', name: 'XGBoost' },
      { key: 'lightgbm', name: 'LightGBM' },
      { key: 'svm', name: 'SVM' },
      { key: 'knn', name: 'KNN' }
    ],
    regression: [
      { key: 'linear_regression', name: '线性回归' },
      { key: 'ridge', name: '岭回归' },
      { key: 'lasso', name: 'Lasso回归' },
      { key: 'random_forest_regressor', name: '随机森林回归' },
      { key: 'xgboost_regressor', name: 'XGBoost回归' },
      { key: 'lightgbm_regressor', name: 'LightGBM回归' }
    ],
    clustering: [
      { key: 'kmeans', name: 'K-Means' },
      { key: 'dbscan', name: 'DBSCAN' },
      { key: 'agglomerative', name: '层次聚类' }
    ],
    time_series: [
      { key: 'arima', name: 'ARIMA' }
    ]
  }
  availableModels.value = modelMap[trainForm.value.taskType] || []
  if (availableModels.value.length > 0) {
    trainForm.value.modelKey = availableModels.value[0].key
    onModelChange()
  }
  trainForm.value.params = {}
  modelParams.value = []
}

function onModelChange() {
  const modelKey = trainForm.value.modelKey
  const paramConfigs = {
    random_forest: [
      { name: 'n_estimators', label: '树的数量', type: 'number', min: 10, max: 500, step: 10, default: 100, hint: '' },
      { name: 'max_depth', label: '最大深度', type: 'number', min: 1, max: 50, step: 1, default: null, hint: 'None表示不限' }
    ],
    xgboost: [
      { name: 'n_estimators', label: '迭代次数', type: 'number', min: 10, max: 500, step: 10, default: 100, hint: '' },
      { name: 'learning_rate', label: '学习率', type: 'number', min: 0.01, max: 1.0, step: 0.01, default: 0.1, hint: '' },
      { name: 'max_depth', label: '最大深度', type: 'number', min: 1, max: 15, step: 1, default: 6, hint: '' }
    ],
    lightgbm: [
      { name: 'n_estimators', label: '迭代次数', type: 'number', min: 10, max: 500, step: 10, default: 100, hint: '' },
      { name: 'learning_rate', label: '学习率', type: 'number', min: 0.01, max: 1.0, step: 0.01, default: 0.1, hint: '' },
      { name: 'num_leaves', label: '叶子节点数', type: 'number', min: 2, max: 255, step: 1, default: 31, hint: '' }
    ],
    kmeans: [
      { name: 'n_clusters', label: '聚类数量 K', type: 'number', min: 2, max: 20, step: 1, default: 3, hint: '' }
    ],
    arima: [
      { name: 'p', label: 'AR阶数', type: 'number', min: 0, max: 5, step: 1, default: 1, hint: '' },
      { name: 'd', label: '差分阶数', type: 'number', min: 0, max: 2, step: 1, default: 1, hint: '' },
      { name: 'q', label: 'MA阶数', type: 'number', min: 0, max: 5, step: 1, default: 1, hint: '' }
    ]
  }

  const configs = paramConfigs[modelKey] || []
  modelParams.value = configs.map(p => ({
    ...p,
    default: p.default !== undefined ? p.default : (p.type === 'number' ? 0 : '')
  }))

  const params = {}
  modelParams.value.forEach(p => {
    params[p.name] = p.default
  })
  trainForm.value.params = params
}

function getModelDescription() {
  const model = availableModels.value.find(m => m.key === trainForm.value.modelKey)
  return model ? model.name : ''
}

// ==================== 训练 ====================
async function handleTrain() {
  if (!sessionStore.currentSessionId) {
    ElMessage.warning('请先上传数据')
    return
  }

  training.value = true
  trainProgress.value = 0
  trainStatusMessage.value = '提交训练任务...'
  trainLogs.value = []

  addLog('info', '开始训练', '提交任务到后端...')

  try {
    const result = await modelsApi.train({
      session_id: sessionStore.currentSessionId,
      task_type: trainForm.value.taskType,
      model_key: trainForm.value.modelKey,
      target_column: trainForm.value.targetColumn || undefined,
      features: trainForm.value.features,
      params: trainForm.value.params,
      user_model_name: trainForm.value.userModelName || undefined
    })

    const taskId = result.task_id
    addLog('info', '任务已提交', `任务ID: ${taskId}`)

    let attempts = 0
    while (attempts < 120) {
      const status = await modelsApi.getTrainStatus(taskId)
      trainProgress.value = status.progress || 0
      trainStatusMessage.value = status.message || '训练中...'

      if (status.status === 'completed') {
        addLog('success', '训练完成', `模型: ${status.user_model_name || status.model_key}`)
        ElMessage.success('训练完成！')
        await loadModels()
        setTimeout(() => {
          trainDialogVisible.value = false
          training.value = false
        }, 1000)
        break
      }
      if (status.status === 'failed') {
        addLog('error', '训练失败', status.message || '未知错误')
        ElMessage.error(status.message || '训练失败')
        training.value = false
        break
      }
      attempts++
      await new Promise(resolve => setTimeout(resolve, 1000))
    }
  } catch (err) {
    addLog('error', '训练异常', err.message)
    ElMessage.error('训练失败: ' + err.message)
    training.value = false
  }
}

function addLog(type, title, message) {
  const time = new Date().toLocaleTimeString()
  trainLogs.value.push({
    time,
    type,
    title,
    message
  })
}

function formatProgress(percentage) {
  return `${percentage}%`
}

// ==================== 预测 ====================
function onModelSelect() {
  const model = savedModels.value.find(
    m => m.model_key === predictForm.value.modelKey
  )
  if (!model) return

  const config = model.config || {}
  const features = config.features || []

  selectedModel.value = {
    ...model,
    features: features
  }

  predictForm.value.inputValues = {}
  features.forEach(f => {
    predictForm.value.inputValues[f] = ''
  })
}

function formatMetrics(metrics) {
  if (!metrics) return '无'
  const parts = []
  if (metrics.accuracy !== undefined) parts.push(`准确率: ${(metrics.accuracy * 100).toFixed(2)}%`)
  if (metrics.r2 !== undefined) parts.push(`R²: ${metrics.r2.toFixed(3)}`)
  if (metrics.f1_score !== undefined) parts.push(`F1: ${metrics.f1_score.toFixed(3)}`)
  return parts.join(' | ') || '无'
}

async function handlePredict() {
  predicting.value = true
  predictResult.value = null

  try {
    const result = await modelsApi.predict({
      model_key: predictForm.value.modelKey,
      session_id: sessionStore.currentSessionId,
      input_values: predictForm.value.inputValues
    })
    predictResult.value = result
    ElMessage.success('预测完成')
  } catch (err) {
    ElMessage.error('预测失败: ' + err.message)
  } finally {
    predicting.value = false
  }
}
</script>

<style scoped>
.model-center {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}
.subtitle {
  color: #909399;
  margin-bottom: 24px;
}

.train-section,
.predict-section {
  padding: 20px 0;
}

/* ===== 推荐列表 ===== */
.recommendations {
  margin-bottom: 20px;
}
.recommendations h4 {
  margin-bottom: 12px;
  color: #2c3e50;
}
.recommend-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.recommend-card {
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
  padding: 12px;
  border: 2px solid transparent;
}
.recommend-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}
.recommend-card .rec-priority {
  position: absolute;
  top: 8px;
  right: 12px;
  font-size: 16px;
}
.recommend-card .rec-task {
  font-weight: 600;
  font-size: 14px;
  color: #2c3e50;
  margin-bottom: 4px;
  padding-right: 30px;
}
.recommend-card .rec-model {
  font-size: 13px;
  color: #409eff;
  margin-bottom: 4px;
}
.recommend-card .rec-target {
  font-size: 12px;
  color: #666;
}
.recommend-card .rec-features {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}
.recommend-card .rec-reason {
  font-size: 12px;
  color: #67c23a;
  margin-top: 4px;
  background: #f0f9eb;
  padding: 4px 8px;
  border-radius: 4px;
}
.recommend-card .rec-btn {
  margin-top: 10px;
  width: 100%;
}
.empty-hint {
  padding: 20px;
  text-align: center;
  color: #909399;
  background: #f5f7fa;
  border-radius: 8px;
}

.hint {
  font-size: 12px;
  color: #909399;
  margin-left: 12px;
}

/* ===== 弹窗 ===== */
.dialog-body {
  max-height: 70vh;
  overflow-y: auto;
  padding-right: 4px;
}
.dialog-body::-webkit-scrollbar {
  width: 6px;
}
.dialog-body::-webkit-scrollbar-thumb {
  background: #c0c4cc;
  border-radius: 3px;
}
.dialog-body::-webkit-scrollbar-track {
  background: #f0f2f6;
  border-radius: 3px;
}

.dialog-summary {
  margin-bottom: 16px;
}

.params-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px 16px;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 8px;
}
.param-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.param-item label {
  font-size: 12px;
  font-weight: 500;
  color: #2c3e50;
}
.param-item .param-hint {
  font-size: 11px;
  color: #909399;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

/* ===== 弹窗内的训练进度 ===== */
.train-progress-dialog {
  margin-top: 8px;
}

.status-message {
  margin-top: 8px;
  color: #909399;
  font-size: 14px;
}

.train-logs {
  margin-top: 12px;
}
.train-logs h4 {
  margin-bottom: 8px;
  font-size: 13px;
  color: #2c3e50;
}
.log-container {
  max-height: 150px;
  overflow-y: auto;
  background: #1e1e1e;
  border-radius: 6px;
  padding: 8px 12px;
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 12px;
}
.log-line {
  display: flex;
  gap: 12px;
  padding: 2px 0;
  color: #d4d4d4;
}
.log-line .log-time {
  color: #6a9955;
  flex-shrink: 0;
}
.log-line.info .log-content {
  color: #d4d4d4;
}
.log-line.success .log-content {
  color: #4ec9b0;
}
.log-line.error .log-content {
  color: #f44747;
}

/* ===== 预测 ===== */
.model-info {
  margin: 16px 0;
}
.input-fields {
  margin: 16px 0;
}
.predict-result {
  margin-top: 20px;
}
.result-value {
  font-size: 20px;
  font-weight: bold;
  color: #409eff;
}
.prob-dist {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.prob-dist span {
  padding: 2px 10px;
  background: #f0f2f6;
  border-radius: 12px;
  font-size: 12px;
}

.empty-state {
  padding: 40px 0;
}
</style>