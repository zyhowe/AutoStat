<template>
  <div class="start-analysis">
    <!-- 顶部执行按钮 -->
    <div class="execution-bar-top">
      <div class="execution-left">
        <span class="execution-label">📌 当前状态：</span>
        <el-tag v-if="scenarioConfigSaved" type="success" size="small">已配置</el-tag>
        <el-tag v-else type="warning" size="small">未配置</el-tag>
        <span class="execution-info" v-if="scenarioConfigSaved">已勾选 {{ enabledCount }} 个场景</span>
        <span v-if="Object.keys(fieldMapping).length > 0" class="mapping-status">📝 {{ Object.keys(fieldMapping).length }} 个映射</span>
        <span v-else class="mapping-status" style="color: #f56c6c;">⚠️ 未配置映射</span>
      </div>
      <div class="execution-right">
        <el-button
          type="primary"
          size="large"
          :loading="executing"
          :disabled="!scenarioConfigSaved || enabledCount === 0 || Object.keys(fieldMapping).length === 0"
          @click="$emit('run-analysis')"
        >
          {{ executing ? '执行中...' : '🚀 场景生成' }}
        </el-button>
      </div>
    </div>

    <!-- 进度条 -->
    <div v-if="executing" class="execution-progress">
      <el-progress :percentage="execProgress" :format="formatProgress" />
      <p class="progress-message">{{ execMessage }}</p>
    </div>

    <!-- 场景配置 -->
    <div class="config-section">
      <div class="section-header">
        <span class="section-title">📋 场景配置</span>
        <div class="header-actions">
          <el-button size="small" @click="$emit('select-all', true)">✅ 全选</el-button>
          <el-button size="small" @click="$emit('select-all', false)">❌ 全不选</el-button>
          <el-tag size="small" type="info">共 {{ candidates.length }} 个场景</el-tag>
        </div>
      </div>

      <div v-if="candidates.length === 0" class="empty-hint">
        暂无候选场景，请先完成数据分析
      </div>

      <div v-else class="candidate-cards">
        <div
          v-for="scenario in candidates"
          :key="scenario.id"
          class="candidate-card"
          :class="{ disabled: !scenario.enabled }"
        >
          <div class="card-header">
            <div class="header-left">
              <span class="scenario-id">{{ scenario.id }}</span>
              <span class="scenario-name">{{ scenario.name }}</span>
              <span class="category-tag">{{ scenario.category || '通用' }}</span>
            </div>
            <div class="header-right">
              <el-switch v-model="scenario.enabled" size="small" />
            </div>
          </div>
          <div class="card-body">
            <div class="trigger-basis">
              <span class="label">📌 触发依据：</span>
              <span class="value">{{ scenario.trigger_basis || '技术特征满足触发条件' }}</span>
            </div>
            <div class="scenario-desc">
              <span class="label">📋 说明：</span>
              <span class="value">{{ scenario.description || '自动推导的业务场景' }}</span>
            </div>
            <div v-if="Object.keys(scenario.params || {}).length > 0" class="scenario-params">
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
        </div>
      </div>

      <!-- 场景配置底部 -->
      <div class="config-actions">
        <el-button type="primary" :loading="savingScenarios" @click="$emit('save-scenario-config')">
          {{ savingScenarios ? '保存中...' : '💾 保存场景配置' }}
        </el-button>
        <el-button @click="$emit('reset-scenario-config')">重置</el-button>
        <el-button size="small" @click="$emit('select-all', true)">✅ 全选</el-button>
        <el-button size="small" @click="$emit('select-all', false)">❌ 全不选</el-button>
        <el-tag v-if="scenarioConfigSaved" type="success" size="small">✅ 已保存</el-tag>
      </div>
    </div>

    <!-- 字段映射 -->
    <div class="config-section">
      <div class="section-header">
        <span class="section-title">📝 字段映射</span>
        <el-tag size="small" type="info">用于业务解读时替换字段名</el-tag>
      </div>
      <div class="mapping-config">
        <el-alert
          title="输入字段名与中文名的对应关系，支持任意格式"
          type="info"
          show-icon
          :closable="false"
          style="margin-bottom: 12px"
        />
        <div class="mapping-input-row">
          <el-input
            :model-value="mappingText"
            type="textarea"
            :rows="4"
            placeholder="粘贴字段映射信息..."
            style="flex: 1;"
            @update:model-value="$emit('update:mapping-text', $event)"
          />
          <div class="mapping-actions">
            <el-button type="primary" @click="$emit('parse-mapping')" :loading="parsing">🔍 解析映射</el-button>
            <el-button @click="$emit('update:mapping-text', '')">清空</el-button>
            <el-button type="success" plain @click="$emit('load-sample-mapping')">加载示例</el-button>
          </div>
        </div>

        <div v-if="Object.keys(parsedMapping).length > 0 || parsedUnmatched.length > 0" class="mapping-result">
          <el-divider />
          <div class="result-title">解析结果</div>
          <div class="mapping-list">
            <div v-for="(value, key) in parsedMapping" :key="key" class="mapping-item">
              <span class="mapping-key">{{ key }}</span>
              <span class="mapping-arrow">→</span>
              <el-input :model-value="value" size="small" style="width:160px;" @update:model-value="(val) => { parsedMapping[key] = val }" />
              <el-button size="small" text type="danger" @click="$emit('remove-parsed-mapping', key)">✕</el-button>
            </div>
          </div>
          <div v-if="parsedUnmatched.length > 0" class="unmatched-list">
            <div class="unmatched-title">⚠️ 未识别字段：</div>
            <el-tag v-for="item in parsedUnmatched" :key="item" size="small" type="warning" style="margin:2px;">{{ item }}</el-tag>
          </div>
          <div class="mapping-actions-bottom">
            <el-button type="primary" :disabled="Object.keys(parsedMapping).length === 0" @click="$emit('confirm-mapping')">✅ 确认映射</el-button>
            <el-button @click="$emit('clear-parsed')">清空结果</el-button>
          </div>
        </div>

        <div v-if="Object.keys(fieldMapping).length > 0" class="current-mapping">
          <el-divider />
          <div class="result-title">当前已保存的映射（{{ Object.keys(fieldMapping).length }} 个）</div>
          <div class="mapping-list">
            <div v-for="(value, key) in fieldMapping" :key="key" class="mapping-item">
              <span class="mapping-key">{{ key }}</span>
              <span class="mapping-arrow">→</span>
              <span class="mapping-value">{{ value }}</span>
              <el-button size="small" text type="danger" @click="$emit('remove-mapping', key)">✕</el-button>
            </div>
          </div>
          <el-button size="small" type="danger" plain @click="$emit('clear-all-mapping')" style="margin-top:8px;">清空所有映射</el-button>
        </div>

        <div class="config-actions" style="margin-top: 12px;">
          <el-button type="success" :loading="savingMapping" :disabled="Object.keys(fieldMapping).length === 0" @click="$emit('save-mapping-config')">
            {{ savingMapping ? '保存中...' : '📝 保存字段映射' }}
          </el-button>
          <el-tag v-if="mappingConfigSaved" type="success" size="small">✅ 已保存</el-tag>
        </div>
      </div>
    </div>

    <!-- 底部执行按钮 -->
    <div class="execution-bar-bottom">
      <div class="execution-left">
        <span class="execution-label">📌 当前状态：</span>
        <el-tag v-if="scenarioConfigSaved" type="success" size="small">已配置</el-tag>
        <el-tag v-else type="warning" size="small">未配置</el-tag>
        <span class="execution-info" v-if="scenarioConfigSaved">已勾选 {{ enabledCount }} 个场景</span>
        <span v-if="Object.keys(fieldMapping).length > 0" class="mapping-status">📝 {{ Object.keys(fieldMapping).length }} 个映射</span>
        <span v-else class="mapping-status" style="color: #f56c6c;">⚠️ 未配置映射</span>
      </div>
      <div class="execution-right">
        <el-button
          type="primary"
          size="large"
          :loading="executing"
          :disabled="!scenarioConfigSaved || enabledCount === 0 || Object.keys(fieldMapping).length === 0"
          @click="$emit('run-analysis')"
        >
          {{ executing ? '执行中...' : '🚀 场景生成' }}
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  candidates: { type: Array, default: () => [] },
  fieldMapping: { type: Object, default: () => ({}) },
  scenarioConfigSaved: { type: Boolean, default: false },
  mappingConfigSaved: { type: Boolean, default: false },
  executing: { type: Boolean, default: false },
  execProgress: { type: Number, default: 0 },
  execMessage: { type: String, default: '' },
  savingScenarios: { type: Boolean, default: false },
  savingMapping: { type: Boolean, default: false },
  parsing: { type: Boolean, default: false },
  enabledCount: { type: Number, default: 0 },
  mappingText: { type: String, default: '' },
  parsedMapping: { type: Object, default: () => ({}) },
  parsedUnmatched: { type: Array, default: () => [] }
})

const emit = defineEmits([
  'run-analysis',
  'save-scenario-config',
  'reset-scenario-config',
  'save-mapping-config',
  'parse-mapping',
  'confirm-mapping',
  'remove-mapping',
  'clear-all-mapping',
  'load-sample-mapping',
  'select-all',
  'remove-parsed-mapping',
  'clear-parsed',
  'update:mapping-text',
  'update:parsed-mapping',
  'update:parsed-unmatched'
])

const formatProgress = (percentage) => `${percentage}%`
</script>

<style scoped>
.start-analysis { padding: 10px 0; }

.execution-bar-top,
.execution-bar-bottom {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #f0f7ff;
  border-radius: 8px;
  border: 1px solid #d9ecff;
  flex-wrap: wrap;
  gap: 8px;
}
.execution-bar-bottom {
  margin-top: 20px;
  background: #f0fdf4;
  border-color: #bbf7d0;
}
.execution-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.execution-label { font-size: 13px; color: #2c3e50; }
.execution-info { font-size: 12px; color: #909399; }
.mapping-status { font-size: 12px; margin-left: 8px; }
.execution-right { display: flex; gap: 8px; flex-wrap: wrap; }

.execution-progress {
  padding: 16px;
  background: #f5f7fa;
  border-radius: 8px;
  margin: 12px 0 16px 0;
}
.progress-message { margin-top: 8px; font-size: 13px; color: #909399; }

.config-section { margin-bottom: 30px; }
.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.section-title { font-size: 16px; font-weight: 600; color: #2c3e50; }
.header-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.empty-hint { padding: 20px; text-align: center; color: #909399; background: #f5f7fa; border-radius: 8px; }

.candidate-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.candidate-card {
  background: #f8f9fa;
  border-radius: 10px;
  padding: 12px 16px;
  border: 1px solid #e4e7ed;
  transition: all 0.2s;
}
.candidate-card:hover { background: #f0f2f5; border-color: #409eff; }
.candidate-card.disabled { opacity: 0.5; }
.candidate-card .card-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 4px; }
.candidate-card .header-left { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.candidate-card .scenario-id { font-size: 11px; font-weight: 600; color: #909399; background: #e9ecef; padding: 2px 10px; border-radius: 4px; }
.candidate-card .scenario-name { font-size: 14px; font-weight: 600; color: #2c3e50; }
.candidate-card .category-tag { font-size: 11px; color: #fff; background: #409eff; padding: 2px 10px; border-radius: 12px; }
.candidate-card .card-body { margin-top: 6px; font-size: 12px; color: #555; line-height: 1.5; }
.candidate-card .card-body .label { color: #909399; }
.candidate-card .card-body .value { color: #2c3e50; }
.candidate-card .card-body .trigger-basis, .candidate-card .card-body .scenario-desc { padding: 2px 0; }
.candidate-card .card-body .scenario-params { margin-top: 4px; }

.config-actions {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 12px;
}

.mapping-config { background: #f8f9fa; padding: 16px; border-radius: 8px; border: 1px solid #e4e7ed; }
.mapping-input-row { display: flex; gap: 12px; align-items: flex-start; }
.mapping-actions { display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
.mapping-result { margin-top: 12px; }
.result-title { font-size: 14px; font-weight: 600; color: #2c3e50; margin-bottom: 8px; }
.mapping-list { display: flex; flex-direction: column; gap: 6px; }
.mapping-item { display: flex; align-items: center; gap: 10px; padding: 6px 10px; background: #fff; border-radius: 6px; border: 1px solid #e4e7ed; flex-wrap: wrap; }
.mapping-key { font-family: 'Consolas', monospace; font-size: 13px; font-weight: 500; color: #2c3e50; min-width: 100px; }
.mapping-arrow { color: #909399; }
.mapping-value { font-size: 13px; color: #409eff; }
.unmatched-list { margin-top: 8px; }
.unmatched-title { font-size: 12px; color: #909399; margin-bottom: 4px; }
.mapping-actions-bottom { margin-top: 12px; display: flex; gap: 8px; }
.current-mapping { margin-top: 12px; }

@media (max-width: 768px) {
  .execution-bar-top,
  .execution-bar-bottom {
    flex-direction: column;
    align-items: stretch;
  }
  .execution-right { justify-content: flex-end; }
  .candidate-cards { grid-template-columns: 1fr; }
  .section-header { flex-direction: column; align-items: flex-start; }
  .mapping-input-row { flex-direction: column; }
  .mapping-actions { flex-direction: row; }
}
</style>