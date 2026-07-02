<template>
  <div class="export-buttons">
    <el-button
      v-for="btn in buttons"
      :key="btn.format"
      :type="btn.type"
      :loading="loading === btn.format"
      @click="handleExport(btn.format)"
    >
      {{ btn.icon }} {{ btn.label }}
    </el-button>
  </div>
</template>

<script setup>
import { ref, defineProps, defineEmits } from 'vue'

const props = defineProps({
  sessionId: {
    type: String,
    required: true
  },
  formats: {
    type: Array,
    default: () => ['html', 'json', 'excel']
  }
})

const emit = defineEmits(['export'])

const loading = ref(null)

const buttonConfig = {
  html: { label: '导出 HTML', icon: '📄', type: 'primary' },
  json: { label: '导出 JSON', icon: '📋', type: 'success' },
  excel: { label: '导出 Excel', icon: '📊', type: 'warning' }
}

const buttons = props.formats.map(f => ({
  format: f,
  ...buttonConfig[f]
}))

async function handleExport(format) {
  loading.value = format
  try {
    await emit('export', format)
  } finally {
    loading.value = null
  }
}
</script>

<style scoped>
.export-buttons {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
</style>