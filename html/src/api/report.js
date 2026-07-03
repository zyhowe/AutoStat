import api from './index'

export const reportApi = {
  get(sessionId) {
    return api.get(`/report/${sessionId}`)
  },

  getSummary(sessionId) {
    return api.get(`/report/${sessionId}/summary`)
  },

  getInsights(sessionId) {
    return api.get(`/report/${sessionId}/insights`)
  },

  getQuality(sessionId) {
    return api.get(`/quality/${sessionId}`)
  },

  export(sessionId, format = 'html') {
    return api.get(`/export/${sessionId}`, {
      params: { format },
      responseType: 'blob'
    })
  },

  // 🆕 导出日志
  exportLog(sessionId) {
    return api.get(`/export/${sessionId}/log`, {
      responseType: 'blob'
    })
  }
}