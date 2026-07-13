// src/components/DataPreviewDialog.js
import { usePreviewStore } from '../stores/preview'

/**
 * 打开数据预览弹窗
 *
 * @param {Object} options
 * @param {string} options.sessionId - 会话 ID
 * @param {string} options.title - 弹窗标题
 * @param {Array} options.filters - 筛选条件 [{ field, condition, value }]
 * @param {Array} options.fields - 要显示的字段列表（可选）
 */
export function openDataPreview(options) {
  const previewStore = usePreviewStore()
  previewStore.open(options)
}

export function closeDataPreview() {
  const previewStore = usePreviewStore()
  previewStore.close()
}

// 挂载到全局，方便在非 Vue 上下文中使用
if (typeof window !== 'undefined') {
  window.openDataPreview = openDataPreview
  window.closeDataPreview = closeDataPreview
}