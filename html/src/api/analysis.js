import api from './index'

export const analysisApi = {
  // 执行分析
  run(data) {
    return api.post('/analysis/run', data)
  },

  // 获取分析状态
  getStatus(taskId) {
    return api.get(`/analysis/status/${taskId}`)
  }
}