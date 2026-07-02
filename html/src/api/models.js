import api from './index'

export const modelsApi = {
  // 训练模型
  train(data) {
    return api.post('/models/train', data)
  },

  // 获取训练状态
  getTrainStatus(taskId) {
    return api.get(`/models/train/status/${taskId}`)
  },

  // 列出模型
  list(sessionId) {
    return api.get('/models/list', { params: { session_id: sessionId } })
  },

  // 预测
  predict(data) {
    return api.post('/models/predict', data)
  },

  // 删除模型
  delete(modelKey, sessionId) {
    return api.delete(`/models/${modelKey}`, { params: { session_id: sessionId } })
  },

  // 获取预警规则
  getAlertRules() {
    return api.get('/models/alert/rules')
  },

  // 检查预警
  checkAlert(data) {
    return api.post('/models/alert/check', data)
  }
}