import api from './index'

export const chatApi = {
  // 非流式对话
  chat(sessionId, question, context) {
    return api.post('/chat', { session_id: sessionId, question, context })
  },

  // 流式对话
  chatStream(sessionId, question, context, onChunk, onComplete, onError) {
    const url = `/api/v1/chat/stream`
    const body = JSON.stringify({ session_id: sessionId, question, context })

    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    })
      .then(response => {
        if (!response.ok) throw new Error('请求失败')
        const reader = response.body.getReader()
        const decoder = new TextDecoder()

        function read() {
          reader.read().then(({ done, value }) => {
            if (done) {
              if (onComplete) onComplete()
              return
            }
            const chunk = decoder.decode(value)
            const lines = chunk.split('\n')
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
                } catch (e) {
                  // 忽略解析错误
                }
              }
            }
            read()
          }).catch(err => {
            if (onError) onError(err.message)
          })
        }
        read()
      })
      .catch(err => {
        if (onError) onError(err.message)
      })
  },

  // 获取场景推荐
  getScenarios(sessionId) {
    return api.get('/chat/scenarios', { params: { session_id: sessionId } })
  },

  // 获取推荐问题
  getRecommendedQuestions(sessionId) {
    return api.get('/chat/recommended_questions', { params: { session_id: sessionId } })
  }
}