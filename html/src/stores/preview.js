// src/stores/preview.js
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const usePreviewStore = defineStore('preview', () => {
  const visible = ref(false)
  const sessionId = ref('')
  const title = ref('数据预览')
  const filters = ref([])
  const fields = ref(null)

  function open(options) {
    sessionId.value = options.sessionId || localStorage.getItem('lastSessionId') || ''
    title.value = options.title || '数据预览'
    filters.value = options.filters || []
    fields.value = options.fields || null
    visible.value = true
  }

  function close() {
    visible.value = false
    // 延迟清空数据，避免弹窗关闭时闪烁
    setTimeout(() => {
      filters.value = []
      fields.value = null
    }, 200)
  }

  return {
    visible,
    sessionId,
    title,
    filters,
    fields,
    open,
    close
  }
})