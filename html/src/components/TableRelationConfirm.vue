<template>
  <div class="relation-confirm">
    <!-- ===== 已确认状态（只读） ===== -->
    <div v-if="isConfirmed" class="confirmed-state">
      <el-alert
        :title="localRelations.length > 0 ? '✅ 表间关系已确认' : '✅ 已跳过关系配置'"
        :type="localRelations.length > 0 ? 'success' : 'info'"
        show-icon
        :closable="false"
        style="margin-bottom: 16px"
      >
        <template #default>
          <span v-if="localRelations.length > 0">已确认 {{ localRelations.length }} 条表间关系</span>
          <span v-else>未配置表间关系，将作为独立表进行分析</span>
          <el-button size="small" text type="primary" @click="startEdit" style="margin-left: 12px;">
            🔄 重新编辑
          </el-button>
        </template>
      </el-alert>

      <!-- 只读关系列表 -->
      <el-table v-if="localRelations.length > 0" :data="localRelations" border size="small" max-height="300">
        <el-table-column prop="from_table" label="源表" width="150" />
        <el-table-column prop="from_col" label="源列" width="120" />
        <el-table-column label="→" width="40" align="center" />
        <el-table-column prop="to_table" label="目标表" width="150" />
        <el-table-column prop="to_col" label="目标列" width="120" />
        <el-table-column prop="relation_type" label="关系类型" width="120" align="center">
          <template #default="{ row }">
            <el-tag :type="getRelationTypeTag(row.relation_type)" size="small">
              {{ getRelationTypeLabel(row.relation_type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="100" align="center">
          <template #default="{ row }">
            <el-progress
              :percentage="Math.round(row.confidence * 100)"
              :stroke-width="6"
              :show-text="true"
            />
          </template>
        </el-table-column>
        <el-table-column label="状态" width="80" align="center">
          <template #default>
            <el-tag type="success" size="small">✅ 已确认</el-tag>
          </template>
        </el-table-column>
      </el-table>
      <div v-else class="empty-relations">
        <el-empty description="未配置任何表间关系" :image-size="40" />
      </div>
    </div>

    <!-- ===== 未确认状态（编辑模式） ===== -->
    <div v-else>
      <el-alert
        :title="relations.length > 0 ? '🔗 自动识别到表间关系，请确认后继续分析' : '🔗 未自动发现表间关系，可手动添加'"
        :type="relations.length > 0 ? 'info' : 'warning'"
        show-icon
        :closable="false"
        style="margin-bottom: 16px"
      />

      <!-- 关系列表 -->
      <el-table :data="localRelations" border size="small" max-height="300">
        <el-table-column prop="from_table" label="源表" width="150" />
        <el-table-column prop="from_col" label="源列" width="120" />
        <el-table-column label="→" width="40" align="center" />
        <el-table-column prop="to_table" label="目标表" width="150" />
        <el-table-column prop="to_col" label="目标列" width="120" />
        <el-table-column prop="relation_type" label="关系类型" width="120" align="center">
          <template #default="{ row }">
            <el-tag :type="getRelationTypeTag(row.relation_type)" size="small">
              {{ getRelationTypeLabel(row.relation_type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="100" align="center">
          <template #default="{ row }">
            <el-progress
              :percentage="Math.round(row.confidence * 100)"
              :stroke-width="6"
              :show-text="true"
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100" align="center">
          <template #default="{ row }">
            <el-button
              size="small"
              type="danger"
              text
              @click="removeRelation(row)"
            >
              移除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 空状态 -->
      <div v-if="localRelations.length === 0" class="empty-relations">
        <el-empty description="暂无表间关系，点击下方按钮手动添加" :image-size="40" />
      </div>

      <div class="relation-actions">
        <el-button size="small" type="primary" @click="openAddDialog">
          ➕ 手动添加关系
        </el-button>
        <el-button size="small" @click="resetRelations">
          🔄 重置
        </el-button>
      </div>

      <div class="confirm-actions">
        <el-button type="primary" size="large" @click="handleConfirm">
          ✅ {{ localRelations.length > 0 ? '确认关系，继续分析' : '确认无关系，继续分析' }}
        </el-button>
        <el-button size="large" @click="handleSkip">
          ⏭️ 跳过关系配置
        </el-button>
      </div>
    </div>

    <!-- ===== 手动添加关系弹窗（带字段下拉联动） ===== -->
    <el-dialog v-model="addDialogVisible" title="添加表间关系" width="650px" destroy-on-close>
      <el-form :model="newRelation" label-width="100px">
        <!-- 源表 -->
        <el-form-item label="源表">
          <el-select
            v-model="newRelation.from_table"
            placeholder="请选择源表"
            style="width: 100%"
            @change="onFromTableChange"
          >
            <el-option
              v-for="name in tableNames"
              :key="name"
              :label="name"
              :value="name"
            />
          </el-select>
        </el-form-item>

        <!-- 源列（联动） -->
        <el-form-item label="源列">
          <el-select
            v-model="newRelation.from_col"
            placeholder="请选择源列"
            style="width: 100%"
            :disabled="!newRelation.from_table"
          >
            <el-option
              v-for="col in fromColOptions"
              :key="col"
              :label="col"
              :value="col"
            />
          </el-select>
        </el-form-item>

        <div style="text-align: center; font-size: 20px; color: #909399; margin: 8px 0;">⬇</div>

        <!-- 目标表 -->
        <el-form-item label="目标表">
          <el-select
            v-model="newRelation.to_table"
            placeholder="请选择目标表"
            style="width: 100%"
            @change="onToTableChange"
          >
            <el-option
              v-for="name in tableNames"
              :key="name"
              :label="name"
              :value="name"
            />
          </el-select>
        </el-form-item>

        <!-- 目标列（联动） -->
        <el-form-item label="目标列">
          <el-select
            v-model="newRelation.to_col"
            placeholder="请选择目标列"
            style="width: 100%"
            :disabled="!newRelation.to_table"
          >
            <el-option
              v-for="col in toColOptions"
              :key="col"
              :label="col"
              :value="col"
            />
          </el-select>
        </el-form-item>

        <!-- 关系类型 -->
        <el-form-item label="关系类型">
          <el-select v-model="newRelation.relation_type" style="width: 100%">
            <el-option label="一对一" value="one_to_one" />
            <el-option label="一对多" value="one_to_many" />
            <el-option label="多对一" value="many_to_one" />
          </el-select>
        </el-form-item>

        <!-- 置信度 -->
        <el-form-item label="置信度">
          <el-slider v-model="newRelation.confidence" :min="0.5" :max="1" :step="0.05" />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="addDialogVisible = false">取消</el-button>
        <el-button type="primary" :disabled="!canAddRelation" @click="confirmAddRelation">
          确认添加
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'

const props = defineProps({
  relations: {
    type: Array,
    default: () => []
  },
  tableNames: {
    type: Array,
    default: () => []
  },
  // 新增：表字段映射 { 表名: [字段1, 字段2, ...] }
  tableColumns: {
    type: Object,
    default: () => ({})
  },
  confirmed: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['confirm', 'skip', 'edit'])

const localRelations = ref([])
const addDialogVisible = ref(false)
const isConfirmed = ref(false)

// 弹窗中的新关系
const newRelation = ref({
  from_table: '',
  from_col: '',
  to_table: '',
  to_col: '',
  relation_type: 'many_to_one',
  confidence: 0.8,
  auto_discovered: false
})

// 联动选项
const fromColOptions = ref([])
const toColOptions = ref([])

// 同步 props.confirmed
watch(() => props.confirmed, (val) => {
  isConfirmed.value = val
}, { immediate: true })

// 同步 props.relations
watch(() => props.relations, (newVal) => {
  if (newVal && newVal.length > 0 && localRelations.value.length === 0) {
    localRelations.value = JSON.parse(JSON.stringify(newVal))
  }
}, { immediate: true, deep: true })

// 是否可以添加关系
const canAddRelation = computed(() => {
  return newRelation.value.from_table &&
         newRelation.value.from_col &&
         newRelation.value.to_table &&
         newRelation.value.to_col
})

function getRelationTypeTag(type) {
  const map = {
    'one_to_one': 'info',
    'one_to_many': 'warning',
    'many_to_one': 'success'
  }
  return map[type] || 'info'
}

function getRelationTypeLabel(type) {
  const map = {
    'one_to_one': '一对一',
    'one_to_many': '一对多',
    'many_to_one': '多对一'
  }
  return map[type] || type
}

function removeRelation(row) {
  const index = localRelations.value.indexOf(row)
  if (index > -1) {
    localRelations.value.splice(index, 1)
  }
}

function openAddDialog() {
  addDialogVisible.value = true
  // 重置表单
  newRelation.value = {
    from_table: props.tableNames[0] || '',
    from_col: '',
    to_table: props.tableNames[1] || '',
    to_col: '',
    relation_type: 'many_to_one',
    confidence: 0.8,
    auto_discovered: false
  }
  // 更新下拉选项
  onFromTableChange()
  onToTableChange()
}

// 源表变化时更新源列下拉
function onFromTableChange() {
  const table = newRelation.value.from_table
  fromColOptions.value = props.tableColumns[table] || []
  // 如果当前选中的列不在新选项列表中，清空
  if (newRelation.value.from_col && !fromColOptions.value.includes(newRelation.value.from_col)) {
    newRelation.value.from_col = ''
  }
}

// 目标表变化时更新目标列下拉
function onToTableChange() {
  const table = newRelation.value.to_table
  toColOptions.value = props.tableColumns[table] || []
  if (newRelation.value.to_col && !toColOptions.value.includes(newRelation.value.to_col)) {
    newRelation.value.to_col = ''
  }
}

function confirmAddRelation() {
  if (!canAddRelation.value) {
    ElMessage.warning('请完整填写所有字段')
    return
  }

  // 检查是否已存在相同关系
  const exists = localRelations.value.some(r =>
    r.from_table === newRelation.value.from_table &&
    r.from_col === newRelation.value.from_col &&
    r.to_table === newRelation.value.to_table &&
    r.to_col === newRelation.value.to_col
  )
  if (exists) {
    ElMessage.warning('该关系已存在')
    return
  }

  // 检查是否反向存在
  const reverseExists = localRelations.value.some(r =>
    r.from_table === newRelation.value.to_table &&
    r.from_col === newRelation.value.to_col &&
    r.to_table === newRelation.value.from_table &&
    r.to_col === newRelation.value.from_col
  )
  if (reverseExists) {
    ElMessage.warning('已存在反向关系，无需重复添加')
    return
  }

  localRelations.value.push({
    ...newRelation.value,
    auto_discovered: false
  })

  addDialogVisible.value = false
  ElMessage.success('关系已添加')
}

function resetRelations() {
  if (props.relations && props.relations.length > 0) {
    localRelations.value = JSON.parse(JSON.stringify(props.relations))
  } else {
    localRelations.value = []
  }
  ElMessage.success('已重置')
}

function handleConfirm() {
  isConfirmed.value = true
  emit('confirm', localRelations.value)
}

function handleSkip() {
  isConfirmed.value = true
  emit('confirm', [])
}

function startEdit() {
  isConfirmed.value = false
  emit('edit')
}

defineExpose({
  isConfirmed,
  localRelations
})
</script>

<style scoped>
.relation-confirm {
  padding: 16px;
  background: #fafafa;
  border-radius: 8px;
  margin: 16px 0;
}

.confirmed-state {
  border: 1px solid #e1f3d8;
  border-radius: 8px;
  padding: 4px 4px 12px 4px;
  background: #fafffe;
}

.empty-relations {
  padding: 16px 0;
}

.relation-actions {
  margin: 12px 0;
  display: flex;
  gap: 8px;
}

.confirm-actions {
  margin: 16px 0;
  display: flex;
  gap: 12px;
}

:deep(.el-dialog) {
  border-radius: 12px;
}
</style>