import { computed } from 'vue'
import { useAnalysisStore } from '../stores/analysis'

export function useAnalysisData() {
  const store = useAnalysisStore()

  const reportData = computed(() => store.reportData)
  const currentTable = computed({
    get: () => store.currentTable || 'merged',
    set: (val) => { store.currentTable = val }
  })

  const allTables = computed(() => reportData.value?.all_tables || {})
  const tableNames = computed(() => reportData.value?.table_names || [])
  const isMultiTable = computed(() => reportData.value?.is_multi_table || false)

  const currentData = computed(() => {
    if (!allTables.value) return {}
    return allTables.value[currentTable.value] || allTables.value['merged'] || {}
  })

  // 快捷访问
  const dataShape = computed(() => currentData.value?.data_shape || { rows: 0, columns: 0 })
  const variableTypes = computed(() => currentData.value?.variable_types || {})
  const variableSummaries = computed(() => currentData.value?.variable_summaries || {})
  const qualityReport = computed(() => currentData.value?.quality_report || {})
  const correlations = computed(() => currentData.value?.correlations || {})
  const timeSeriesDiagnostics = computed(() => currentData.value?.time_series_diagnostics || {})
  const modelRecommendations = computed(() => currentData.value?.model_recommendations || [])
  const cleaningSuggestions = computed(() => currentData.value?.cleaning_suggestions || [])
  const columnNames = computed(() => currentData.value?.column_names || [])

  return {
    // 原始数据
    reportData,
    allTables,
    currentTable,
    tableNames,
    isMultiTable,
    currentData,
    // 快捷访问
    dataShape,
    variableTypes,
    variableSummaries,
    qualityReport,
    correlations,
    timeSeriesDiagnostics,
    modelRecommendations,
    cleaningSuggestions,
    columnNames,
    // 方法
    setCurrentTable: (val) => { store.currentTable = val }
  }
}