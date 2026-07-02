import api from './index'

export const sessionApi = {
  // 创建会话
  create(data) {
    return api.post('/session/create', data)
  },
  
  // 获取项目列表
  list() {
    return api.get('/session/list')
  },
  
  // 获取会话信息
  get(sessionId) {
    return api.get(`/session/${sessionId}`)
  },
  
  // 删除会话
  delete(sessionId) {
    return api.delete(`/session/${sessionId}`)
  }
}