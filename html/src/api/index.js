// src/api/index.js
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://10.17.181.188:8000/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 🚫 删除 X-Client-IP 请求头设置
api.interceptors.response.use(
  response => response.data,
  error => {
    const message = error.response?.data?.detail || error.message || '请求失败'
    return Promise.reject({ message, status: error.response?.status })
  }
)

export default api