import api from './index'

export const configApi = {
  // 数据库配置
  getDatabase() {
    return api.get('/config/database')
  },
  saveDatabase(data) {
    return api.post('/config/database', data)
  },
  deleteDatabase(name) {
    return api.delete(`/config/database/${name}`)
  },
  // ✅ 新增：测试数据库连接
  testDatabase(data) {
    return api.post('/config/database/test', data)
  },

  // 大模型配置
  getLLM() {
    return api.get('/config/llm')
  },
  saveLLM(data) {
    return api.post('/config/llm', data)
  },
  deleteLLM(name) {
    return api.delete(`/config/llm/${name}`)
  },
  testLLM(data) {
    return api.post('/config/llm/test', data)
  }
}