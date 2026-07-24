// src/api/stream.js
/**
 * 流式查询 API - 独立模块
 * 用于全量数据解码，支持从 SQL Server 流式读取数据
 */

/**
 * 流式查询原始数据
 * @param {Object} options
 * @param {string} options.sessionId - 会话ID
 * @param {Object} options.context - 追溯上下文
 * @param {number} options.batchSize - 批大小，默认 100
 * @param {number} options.maxRows - 最大行数，默认 10000
 * @param {Function} options.onChunk - (row, count)
 * @param {Function} options.onMeta - (columns, totalEstimate)
 * @param {Function} options.onComplete - (rowCount, cancelled)
 * @param {Function} options.onError - (message, isWarning)
 * @param {Function} options.onInfo - (description, sql)
 * @param {AbortController} options.abortController
 */
export async function streamQuery({
  sessionId,
  context,
  batchSize = 100,
  maxRows = 10000,
  onChunk,
  onMeta,
  onComplete,
  onError,
  onInfo,
  abortController
}) {
  const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://10.17.181.188:8000/api/v1'
  const url = `${baseURL}/data/stream-query`

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        context: context,
        batch_size: batchSize,
        max_rows: maxRows
      }),
      signal: abortController?.signal
    })

    if (!response.ok) {
      const errorText = await response.text()
      let errorMsg = `HTTP ${response.status}`
      try {
        const errorJson = JSON.parse(errorText)
        errorMsg = errorJson.detail || errorMsg
      } catch {
        errorMsg = errorText || errorMsg
      }
      if (onError) onError(errorMsg)
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    let rowCount = 0

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const data = JSON.parse(line)

          if (data.type === 'meta') {
            if (onMeta) onMeta(data.columns, data.total_estimate)
          } else if (data.type === 'complete') {
            if (onComplete) onComplete(data.row_count, false)
          } else if (data.type === 'error') {
            if (onError) onError(data.message, false)
          } else if (data.type === 'warning') {
            if (onError) onError(data.message, true)
          } else if (data.type === 'info') {
            if (onInfo) onInfo(data.description, data.sql)
          } else {
            rowCount++
            if (onChunk) onChunk(data, rowCount)
          }
        } catch (e) {
          console.warn('[流式查询] 解析失败:', line, e)
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      if (onComplete) onComplete(0, true)
    } else {
      if (onError) onError(err.message || '连接失败', false)
    }
  }
}

/**
 * 预览（仅前 10 行）
 */
export async function streamPreview(options) {
  return streamQuery({
    ...options,
    batchSize: 10,
    maxRows: 10
  })
}

export const streamApi = {
  streamQuery,
  streamPreview
}

export default streamApi