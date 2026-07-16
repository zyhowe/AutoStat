<template>
  <div class="relation-confirm">
    <!-- ===== 已确认状态（只读） ===== -->
    <div v-if="isConfirmed" class="confirmed-state">
      <el-alert
        title="✅ 表间关系已确认"
        type="success"
        show-icon
        :closable="false"
        style="margin-bottom: 16px"
      >
        <template #default>
          <span>已确认 {{ localRelations.length }} 条表间关系</span>
          <el-button size="small" text type="primary" @click="startEdit" style="margin-left: 12px;">
            🔄 重新编辑
          </el-button>
        </template>
      </el-alert>

      <!-- 只读关系列表 -->
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
        <el-table-column label="状态" width="80" align="center">
          <template #default>
            <el-tag type="success" size="small">✅ 已确认</el-tag>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- ===== 未确认状态（编辑模式） ===== -->
    <div v-else>
      <el-alert
        title="🔗 自动识别到表间关系，请确认后继续分析"
        type="info"
        show-icon
        :closable="false"
        style="margin-bottom: 16px"
      />

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

      <div class="relation-actions">
        <el-button size="small" type="primary" @click="addRelation">
          ➕ 手动添加关系
        </el-button>
        <el-button size="small" @click="resetRelations">
          🔄 重置
        </el-button>
      </div>

      <div class="confirm-actions">
        <el-button type="primary" size="large" @click="handleConfirm">
          ✅ 确认关系，继续分析
        </el-button>
        <el-button size="large" @click="handleSkip">
          ⏭️ 跳过关系配置
        </el-button>
      </div>
    </div>

    <!-- ===== 手动添加关系弹窗 ===== -->
    <el-dialog v-model="addDialogVisible" title="添加表间关系" width="600px">
      <el-form :model="newRelation" label-width="100px">
        <el-form-item label="源表">
          <el-select v-model="newRelation.from_table" style="width: 100%">
            <el-option v-for="name in tableNames" :key="name" :label="name" :value="name" />
          </el-select>
        </el-form-item>
        <el-form-item label="源列">
          <el-input v-model="newRelation.from_col" placeholder="请输入源列名" />
        </el-form-item>
        <el-form-item label="目标表">
          <el-select v-model="newRelation.to_table" style="width: 100%">
            <el-option v-for="name in tableNames" :key="name" :label="name" :value="name" />
          </el-select>
        </el-form-item>
        <el-form-item label="目标列">
          <el-input v-model="newRelation.to_col" placeholder="请输入目标列名" />
        </el-form-item>
        <el-form-item label="关系类型">
          <el-select v-model="newRelation.relation_type" style="width: 100%">
            <el-option label="一对一" value="one_to_one" />
            <el-option label="一对多" value="one_to_many" />
            <el-option label="多对一" value="many_to_one" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="addDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="confirmAddRelation">确认添加</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
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
  confirmed: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['confirm', 'skip', 'edit'])

const localRelations = ref([])
const addDialogVisible = ref(false)
const isConfirmed = ref(false)

// 同步 props.confirmed
watch(() => props.confirmed, (val) => {
  isConfirmed.value = val
}, { immediate: true })

// 同步 props.relations
watch(() => props.relations, (newVal) => {
  localRelations.value = JSON.parse(JSON.stringify(newVal))
}, { immediate: true })

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

function addRelation() {
  addDialogVisible.value = true
  if (props.tableNames.length >= 2) {
    newRelation.value.from_table = props.tableNames[0]
    newRelation.value.to_table = props.tableNames[1]
  }
}

const newRelation = ref({
  from_table: '',
  from_col: '',
  to_table: '',
  to_col: '',
  relation_type: 'many_to_one',
  confidence: 0.8,
  auto_discovered: false
})

function confirmAddRelation() {
  if (!newRelation.value.from_table || !newRelation.value.from_col ||
      !newRelation.value.to_table || !newRelation.value.to_col) {
    ElMessage.warning('请完整填写所有字段')
    return
  }
  localRelations.value.push({
    ...newRelation.value,
    auto_discovered: false
  })
  addDialogVisible.value = false
  newRelation.value = {
    from_table: '',
    from_col: '',
    to_table: '',
    to_col: '',
    relation_type: 'many_to_one',
    confidence: 0.8,
    auto_discovered: false
  }
  ElMessage.success('关系已添加')
}

function resetRelations() {
  localRelations.value = JSON.parse(JSON.stringify(props.relations))
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
</style>