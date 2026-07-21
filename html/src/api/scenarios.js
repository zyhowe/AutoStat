import api from './index'

export const scenariosApi = {
  // 获取场景列表
  get(sessionId) {
    return api.get(`/scenarios/${sessionId}`)
  },

  // 更新场景
  update(sessionId, scenarios) {
    return api.post(`/scenarios/${sessionId}/update`, { scenarios })
  },

  // 执行场景
  execute(sessionId, scenarioIds = null) {
    return api.post(`/scenarios/${sessionId}/execute`, {
      scenario_ids: scenarioIds
    })
  },

  // 解析字段映射（大模型抽取）
  parseMapping(sessionId, text) {
    return api.post(`/scenarios/${sessionId}/mapping/parse`, { text })
  },

  // 翻译场景
  translate(sessionId, fieldMapping) {
    return api.post(`/scenarios/${sessionId}/translate`, {
      field_mapping: fieldMapping
    })
  },

  // 获取仪表板数据
  getDashboard(sessionId) {
    return api.get(`/scenarios/${sessionId}/dashboard`)
  },

  // 保存字段映射
  saveMapping(sessionId, fieldMapping) {
    return api.post(`/scenarios/${sessionId}/mapping/save`, {
      field_mapping: fieldMapping
    })
  },

  // 获取洞察数据
  getInsights(sessionId) {
    return api.get(`/scenarios/${sessionId}/insights`)
  }
}