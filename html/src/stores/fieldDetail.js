// src/stores/fieldDetail.js
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useFieldDetailStore = defineStore('fieldDetail', () => {
  const visible = ref(false)
  const fieldName = ref('')
  const fieldData = ref({})

  function open(field, data) {
    fieldName.value = field
    fieldData.value = data
    visible.value = true
  }

  function close() {
    visible.value = false
  }

  return { visible, fieldName, fieldData, open, close }
})