<template>
  <div class="settings-page">
    <h2>⚙️ 设置</h2>
    <p class="subtitle">数据库和大模型配置管理</p>

    <el-tabs v-model="activeTab">
      <!-- 大模型配置 -->
      <el-tab-pane label="🤖 大模型配置" name="llm">
        <div class="config-section">
          <el-alert
            title="大模型配置用于AI智能解读和对话功能"
            type="info"
            show-icon
            :closable="false"
            style="margin-bottom: 16px"
          />

          <!-- 已有配置列表 -->
          <div v-if="llmConfigs.length > 0" class="config-list">
            <h4>已有配置</h4>
            <el-table :data="llmConfigs" border style="width: 100%">
              <el-table-column prop="name" label="配置名称" width="150" />
              <el-table-column prop="api_base" label="API地址" />
              <el-table-column prop="model" label="模型名称" width="150" />
              <el-table-column label="操作" width="180">
                <template #default="{ row }">
                  <el-button size="small" type="primary" @click="selectLlmConfig(row)">选择</el-button>
                  <el-button size="small" type="danger" @click="deleteLlmConfig(row.name)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
            <div v-if="selectedLlmName" class="selected-info">
              <el-tag type="success">当前使用: {{ selectedLlmName }}</el-tag>
            </div>
          </div>

          <div v-else class="empty-config">
            <el-empty description="暂无大模型配置" :image-size="60" />
          </div>

          <!-- 添加新配置 -->
          <el-divider />
          <h4>添加新配置</h4>
          <el-form :model="llmForm" label-width="120px" style="max-width: 600px">
            <el-form-item label="配置名称">
              <el-input v-model="llmForm.name" placeholder="例如: DeepSeek, 本地Qwen" />
            </el-form-item>
            <el-form-item label="API地址">
              <el-input v-model="llmForm.api_base" placeholder="https://api.deepseek.com/v1" />
            </el-form-item>
            <el-form-item label="API密钥">
              <el-input v-model="llmForm.api_key" type="password" placeholder="sk-xxx" show-password />
            </el-form-item>
            <el-form-item label="模型名称">
              <el-input v-model="llmForm.model" placeholder="deepseek-chat, qwen-7b" />
            </el-form-item>
            <el-form-item label="超时时间">
              <el-input-number v-model="llmForm.timeout" :min="10" :max="300" />
              <span style="margin-left: 8px; color: #909399; font-size: 12px;">秒</span>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="testLlmConnection" :loading="testingLlm">
                {{ testingLlm ? '测试中...' : '🔌 测试连接' }}
              </el-button>
              <el-button type="success" @click="saveLlmConfig" :loading="savingLlm">
                {{ savingLlm ? '保存中...' : '💾 保存配置' }}
              </el-button>
            </el-form-item>
            <div v-if="llmTestResult" class="test-result">
              <el-alert
                :title="llmTestResult.message"
                :type="llmTestResult.success ? 'success' : 'error'"
                show-icon
                :closable="false"
              />
            </div>
          </el-form>
        </div>
      </el-tab-pane>

      <!-- 数据库配置 -->
      <el-tab-pane label="🗄️ 数据库配置" name="db">
        <div class="config-section">
          <el-alert
            title="数据库配置用于连接SQL Server等数据源"
            type="info"
            show-icon
            :closable="false"
            style="margin-bottom: 16px"
          />

          <!-- 已有配置列表 -->
          <div v-if="dbConfigs.length > 0" class="config-list">
            <h4>已有配置</h4>
            <el-table :data="dbConfigs" border style="width: 100%">
              <el-table-column prop="name" label="配置名称" width="150" />
              <el-table-column prop="server" label="服务器" />
              <el-table-column prop="database" label="数据库" />
              <el-table-column prop="trusted_connection" label="认证方式" width="120">
                <template #default="{ row }">
                  {{ row.trusted_connection ? 'Windows认证' : 'SQL认证' }}
                </template>
              </el-table-column>
              <el-table-column label="操作" width="180">
                <template #default="{ row }">
                  <el-button size="small" type="primary" @click="selectDbConfig(row)">选择</el-button>
                  <el-button size="small" type="danger" @click="deleteDbConfig(row.name)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
            <div v-if="selectedDbName" class="selected-info">
              <el-tag type="success">当前使用: {{ selectedDbName }}</el-tag>
            </div>
          </div>

          <div v-else class="empty-config">
            <el-empty description="暂无数据库配置" :image-size="60" />
          </div>

          <!-- 添加新配置 -->
          <el-divider />
          <h4>添加新配置</h4>
          <el-form :model="dbForm" label-width="120px" style="max-width: 600px">
            <el-form-item label="配置名称">
              <el-input v-model="dbForm.name" placeholder="例如: 生产数据库" />
            </el-form-item>
            <el-form-item label="服务器地址">
              <el-input v-model="dbForm.server" placeholder="localhost 或 IP地址" />
            </el-form-item>
            <el-form-item label="数据库名称">
              <el-input v-model="dbForm.database" placeholder="数据库名" />
            </el-form-item>
            <el-form-item label="用户名">
              <el-input v-model="dbForm.username" placeholder="可选" />
            </el-form-item>
            <el-form-item label="密码">
              <el-input v-model="dbForm.password" type="password" placeholder="可选" show-password />
            </el-form-item>
            <el-form-item label="认证方式">
              <el-switch v-model="dbForm.trusted_connection" />
              <span style="margin-left: 8px; color: #909399; font-size: 12px;">
                {{ dbForm.trusted_connection ? 'Windows身份认证' : 'SQL Server身份认证' }}
              </span>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="testDbConnection" :loading="testingDb">
                {{ testingDb ? '测试中...' : '🔌 测试连接' }}
              </el-button>
              <el-button type="success" @click="saveDbConfig" :loading="savingDb">
                {{ savingDb ? '保存中...' : '💾 保存配置' }}
              </el-button>
            </el-form-item>
            <div v-if="dbTestResult" class="test-result">
              <el-alert
                :title="dbTestResult.message"
                :type="dbTestResult.success ? 'success' : 'error'"
                show-icon
                :closable="false"
              />
            </div>
          </el-form>
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { configApi } from '../api/config'

const activeTab = ref('llm')

// ==================== 大模型配置 ====================
const llmConfigs = ref([])
const selectedLlmName = ref('')
const llmForm = ref({
  name: '',
  api_base: '',
  api_key: '',
  model: '',
  timeout: 60
})
const testingLlm = ref(false)
const savingLlm = ref(false)
const llmTestResult = ref(null)

// ==================== 数据库配置 ====================
const dbConfigs = ref([])
const selectedDbName = ref('')
const dbForm = ref({
  name: '',
  server: '',
  database: '',
  username: '',
  password: '',
  trusted_connection: false
})
const testingDb = ref(false)
const savingDb = ref(false)
const dbTestResult = ref(null)

onMounted(async () => {
  await loadAllConfigs()
})

async function loadAllConfigs() {
  await Promise.all([
    loadLlmConfigs(),
    loadDbConfigs()
  ])
}

async function loadLlmConfigs() {
  try {
    const result = await configApi.getLLM()
    llmConfigs.value = result || []
  } catch (err) {
    console.error('加载大模型配置失败:', err)
  }
}

async function loadDbConfigs() {
  try {
    const result = await configApi.getDatabase()
    dbConfigs.value = result || []
  } catch (err) {
    console.error('加载数据库配置失败:', err)
  }
}

async function testLlmConnection() {
  if (!llmForm.value.api_base || !llmForm.value.model) {
    ElMessage.warning('请填写API地址和模型名称')
    return
  }

  testingLlm.value = true
  llmTestResult.value = null

  try {
    const result = await configApi.testLLM({
      api_base: llmForm.value.api_base,
      api_key: llmForm.value.api_key,
      model: llmForm.value.model
    })
    llmTestResult.value = result
    if (result.success) {
      ElMessage.success('连接成功！')
    } else {
      ElMessage.error('连接失败: ' + result.message)
    }
  } catch (err) {
    llmTestResult.value = { success: false, message: err.message || '测试失败' }
    ElMessage.error('测试失败: ' + err.message)
  } finally {
    testingLlm.value = false
  }
}

async function saveLlmConfig() {
  if (!llmForm.value.name || !llmForm.value.api_base || !llmForm.value.model) {
    ElMessage.warning('请填写完整信息')
    return
  }

  savingLlm.value = true
  try {
    await configApi.saveLLM(llmForm.value)
    ElMessage.success('配置保存成功')
    await loadLlmConfigs()
    // 清空表单
    llmForm.value = { name: '', api_base: '', api_key: '', model: '', timeout: 60 }
    llmTestResult.value = null
  } catch (err) {
    ElMessage.error('保存失败: ' + err.message)
  } finally {
    savingLlm.value = false
  }
}

async function deleteLlmConfig(name) {
  try {
    await ElMessageBox.confirm(`确定要删除配置 "${name}" 吗？`, '确认删除', { type: 'warning' })
    await configApi.deleteLLM(name)
    ElMessage.success('配置已删除')
    if (selectedLlmName.value === name) {
      selectedLlmName.value = ''
    }
    await loadLlmConfigs()
  } catch (err) {
    if (err !== 'cancel') {
      ElMessage.error('删除失败: ' + err.message)
    }
  }
}

function selectLlmConfig(config) {
  selectedLlmName.value = config.name
  ElMessage.success(`已选择配置: ${config.name}`)
}

async function testDbConnection() {
  if (!dbForm.value.server || !dbForm.value.database) {
    ElMessage.warning('请填写服务器和数据库名称')
    return
  }

  testingDb.value = true
  dbTestResult.value = null

  try {
    // 使用数据库连接测试（简化版）
    const result = await configApi.testDatabase(dbForm.value)
    dbTestResult.value = result
    if (result.success) {
      ElMessage.success('连接成功！')
    } else {
      ElMessage.error('连接失败: ' + result.message)
    }
  } catch (err) {
    dbTestResult.value = { success: false, message: err.message || '测试失败' }
    ElMessage.error('测试失败: ' + err.message)
  } finally {
    testingDb.value = false
  }
}

async function saveDbConfig() {
  if (!dbForm.value.name || !dbForm.value.server || !dbForm.value.database) {
    ElMessage.warning('请填写完整信息')
    return
  }

  savingDb.value = true
  try {
    await configApi.saveDatabase(dbForm.value)
    ElMessage.success('配置保存成功')
    await loadDbConfigs()
    dbForm.value = { name: '', server: '', database: '', username: '', password: '', trusted_connection: false }
    dbTestResult.value = null
  } catch (err) {
    ElMessage.error('保存失败: ' + err.message)
  } finally {
    savingDb.value = false
  }
}

async function deleteDbConfig(name) {
  try {
    await ElMessageBox.confirm(`确定要删除配置 "${name}" 吗？`, '确认删除', { type: 'warning' })
    await configApi.deleteDatabase(name)
    ElMessage.success('配置已删除')
    if (selectedDbName.value === name) {
      selectedDbName.value = ''
    }
    await loadDbConfigs()
  } catch (err) {
    if (err !== 'cancel') {
      ElMessage.error('删除失败: ' + err.message)
    }
  }
}

function selectDbConfig(config) {
  selectedDbName.value = config.name
  ElMessage.success(`已选择配置: ${config.name}`)
}
</script>

<style scoped>
.settings-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}
.subtitle {
  color: #909399;
  margin-bottom: 24px;
}
.config-section {
  padding: 10px 0;
}
.config-list {
  margin-bottom: 16px;
}
.config-list h4 {
  margin-bottom: 12px;
  color: #2c3e50;
}
.empty-config {
  padding: 20px 0;
}
.selected-info {
  margin-top: 12px;
}
.test-result {
  margin-top: 12px;
}
</style>