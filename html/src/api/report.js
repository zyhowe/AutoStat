import api from './index'

export const reportApi = {
  // 获取完整报告
  get(sessionId) {
    return api.get(`/report/${sessionId}`)
  },

  // 获取报告摘要
  getSummary(sessionId) {
    return api.get(`/report/${sessionId}/summary`)
  },

  // 获取智能解读
  getInsights(sessionId) {
    return api.get(`/report/${sessionId}/insights`)
  },

  // 获取质量报告
  getQuality(sessionId) {
    return api.get(`/quality/${sessionId}`)
  },

  // 导出报告
  export(sessionId, format = 'html') {
    return api.get(`/export/${sessionId}`, {
      params: { format },
      responseType: 'blob'
    })
  }
}