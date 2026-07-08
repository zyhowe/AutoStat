import api from './index'

export const chatApi = {
  // ===== 非流式对话 =====
  chat(sessionId, question, context = ['json_result']) {
    return api.post('/chat', {
      session_id: sessionId,
      question: question,
      context: context
    })
  },

  // ===== 流式对话（支持 context_data） =====
  chatStream(sessionId, question, context, contextData, onChunk, onComplete, onError) {
    const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://10.17.181.188:8000/api/v1'
    const url = `${baseURL}/chat/stream`
    const body = {
      session_id: sessionId,
      question: question,
      context: context || ['json_result']
    }

    if (contextData) {
      body.context_data = contextData
    }

    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        function read() {
          reader.read().then(({ done, value }) => {
            if (done) {
              if (onComplete) onComplete()
              return
            }

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.substring(6))
                  if (data.done) {
                    if (onComplete) onComplete()
                    return
                  }
                  if (data.content && onChunk) {
                    onChunk(data.content)
                  }
                  if (data.error && onError) {
                    onError(data.error)
                    return
                  }
                } catch (e) {
                  console.debug('SSE parse error:', e)
                }
              }
            }
            read()
          }).catch(err => {
            if (onError) onError(err.message || '读取流失败')
          })
        }
        read()
      })
      .catch(err => {
        if (onError) onError(err.message || '连接失败')
      })
  },

  // ===== 预测流式接口 =====
  predictionStream(sessionId, question, onChunk, onComplete, onError) {
    const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://10.17.181.188:8000/api/v1'
    const url = `${baseURL}/chat/prediction/stream`
    const body = JSON.stringify({
      session_id: sessionId,
      question: question
    })

    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: body
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let resultData = null

        function read() {
          reader.read().then(({ done, value }) => {
            if (done) {
              if (onComplete) onComplete(resultData)
              return
            }

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.substring(6))
                  if (data.content && onChunk) {
                    onChunk(data.content)
                  }
                  if (data.done) {
                    resultData = data.data || data
                    if (onComplete) onComplete(resultData)
                    return
                  }
                  if (data.error && onError) {
                    onError(data.error)
                    return
                  }
                } catch (e) {
                  console.debug('SSE parse error:', e)
                }
              }
            }
            read()
          }).catch(err => {
            if (onError) onError(err.message || '读取流失败')
          })
        }
        read()
      })
      .catch(err => {
        if (onError) onError(err.message || '连接失败')
      })
  },

  // ===== 获取场景推荐 =====
  getScenarios(sessionId) {
    return api.get('/chat/scenarios', {
      params: { session_id: sessionId }
    })
  },

  // ===== 获取推荐问题 =====
  getRecommendedQuestions(sessionId, scene) {
    const params = { session_id: sessionId }
    if (scene) params.scene = scene
    return api.get('/chat/recommended_questions', { params })
  }
}