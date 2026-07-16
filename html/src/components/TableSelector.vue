<template>
  <div class="table-selector-wrapper">
    <span class="selector-label">📋 数据表：</span>
    <el-radio-group v-model="internalValue" size="small" @change="handleChange">
      <el-radio-button
        v-for="item in tableOptions"
        :key="item.value"
        :label="item.value"
      >
        {{ item.label }}
      </el-radio-button>
    </el-radio-group>
    <el-tag v-if="isMultiTable" size="small" type="info" style="margin-left: 12px;">
      共 {{ tableOptions.length }} 个表
    </el-tag>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  modelValue: {
    type: String,
    default: 'merged'
  },
  tableNames: {
    type: Array,
    default: () => []
  },
  isMultiTable: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:modelValue', 'change'])

const internalValue = ref(props.modelValue)

watch(() => props.modelValue, (newVal) => {
  internalValue.value = newVal
})

const tableOptions = computed(() => {
  const options = []
  // 第一个选项始终是合并表
  options.push({
    value: 'merged',
    label: props.isMultiTable ? '📊 合并表' : '📊 数据表'
  })
  // 添加原始表
  for (const name of props.tableNames) {
    options.push({
      value: name,
      label: `📋 ${name}`
    })
  }
  return options
})

function handleChange(value) {
  emit('update:modelValue', value)
  emit('change', value)
}
</script>

<style scoped>
.table-selector-wrapper {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 16px;
  background: #f5f7fa;
  border-radius: 8px;
  margin-bottom: 16px;
}
.selector-label {
  font-size: 13px;
  font-weight: 500;
  color: #2c3e50;
}
</style>