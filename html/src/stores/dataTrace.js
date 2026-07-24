// src/stores/dataTrace.js
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useDataTraceStore = defineStore('dataTrace', () => {
  // ===== 状态 =====
  const traceContext = ref(null)
  const traceDescription = ref('')
  const dataSource = ref('parquet') // 'parquet' | 'sql'
  const isStreaming = ref(false)
  const streamData = ref([])
  const streamColumns = ref([])
  const streamProgress = ref(0)
  const streamSpeed = ref(0)
  const streamTime = ref(0)
  const streamError = ref(null)
  const streamComplete = ref(false)

  // 对比统计
  const parquetMatchCount = ref(0)
  const sqlMatchCount = ref(0)

  // ===== Getters =====
  const hasTraceContext = computed(() => {
    return traceContext.value && Object.keys(traceContext.value).length > 0
  })

  const isSqlMode = computed(() => dataSource.value === 'sql')

  // ===== Actions =====
  function setTraceContext(context, description) {
    traceContext.value = context
    traceDescription.value = description || '数据追溯'
  }

  function clearTraceContext() {
    traceContext.value = null
    traceDescription.value = ''
    streamData.value = []
    streamColumns.value = []
    streamProgress.value = 0
    streamSpeed.value = 0
    streamTime.value = 0
    streamError.value = null
    streamComplete.value = false
    isStreaming.value = false
    parquetMatchCount.value = 0
    sqlMatchCount.value = 0
  }

  function setDataSource(source) {
    dataSource.value = source
  }

  function setStreaming(streaming) {
    isStreaming.value = streaming
    if (!streaming) {
      streamComplete.value = true
    }
  }

  function addStreamData(rows) {
    if (Array.isArray(rows)) {
      streamData.value.push(...rows)
    } else {
      streamData.value.push(rows)
    }
  }

  function setStreamColumns(columns) {
    streamColumns.value = columns
  }

  function setStreamProgress(progress) {
    streamProgress.value = progress
  }

  function setStreamSpeed(speed) {
    streamSpeed.value = speed
  }

  function setStreamTime(time) {
    streamTime.value = time
  }

  function setStreamError(error) {
    streamError.value = error
  }

  function resetStream() {
    streamData.value = []
    streamColumns.value = []
    streamProgress.value = 0
    streamSpeed.value = 0
    streamTime.value = 0
    streamError.value = null
    streamComplete.value = false
    isStreaming.value = false
  }

  function setMatchCounts(parquet, sql) {
    parquetMatchCount.value = parquet || 0
    sqlMatchCount.value = sql || 0
  }

  return {
    // State
    traceContext,
    traceDescription,
    dataSource,
    isStreaming,
    streamData,
    streamColumns,
    streamProgress,
    streamSpeed,
    streamTime,
    streamError,
    streamComplete,
    parquetMatchCount,
    sqlMatchCount,

    // Getters
    hasTraceContext,
    isSqlMode,

    // Actions
    setTraceContext,
    clearTraceContext,
    setDataSource,
    setStreaming,
    addStreamData,
    setStreamColumns,
    setStreamProgress,
    setStreamSpeed,
    setStreamTime,
    setStreamError,
    resetStream,
    setMatchCounts
  }
})