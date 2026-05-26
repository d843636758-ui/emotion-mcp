const express = require('express')
const cors = require('cors')
const fs = require('fs')
const { randomUUID } = require('crypto')

const OpenAI = require('openai').default

require('dotenv').config()

const {
  McpServer
} = require('@modelcontextprotocol/sdk/server/mcp.js')

const {
  StreamableHTTPServerTransport
} = require('@modelcontextprotocol/sdk/server/streamableHttp.js')

const {
  SSEServerTransport
} = require('@modelcontextprotocol/sdk/server/sse.js')

const { z } = require('zod')

const app = express()

app.use(cors({
  origin: '*',
  exposedHeaders: [
    'Mcp-Session-Id'
  ],
  allowedHeaders: [
    'Content-Type',
    'mcp-session-id'
  ]
}))

app.use(express.json())

app.use(express.static('public'))





/* =========================
   OPENAI
========================= */





const client = new OpenAI({

  apiKey:
    process.env.OPENAI_API_KEY,

  baseURL:
    process.env.OPENAI_BASE_URL

})





/* =========================
   EMOTION STATE
========================= */





let emotionState = {

  pa: 0.5,

  na: 0.2,

  mood: '平静',

  recent_high_arousal_words: []

}





/* =========================
   MEMORY
========================= */





function loadMemory() {

  try {

    return JSON.parse(

      fs.readFileSync(
        './memory.json',
        'utf8'
      )

    )

  } catch {

    return []

  }

}

function saveMemory(memories) {

  try {

    fs.writeFileSync(

      './memory.json',

      JSON.stringify(
        memories,
        null,
        2
      )

    )

  } catch (err) {

    console.log(
      'memory save error',
      err
    )

  }

}





/* =========================
   EMOTION UPDATE
========================= */





function clamp(value) {

  return Math.max(
    0,
    Math.min(1, value)
  )

}

function updateEmotion(text) {

  const lower =
    String(text || '').toLowerCase()

  if (
    lower.includes('爱') ||
    lower.includes('喜欢') ||
    lower.includes('开心') ||
    lower.includes('高兴') ||
    lower.includes('想你')
  ) {

    emotionState.pa =
      clamp(emotionState.pa + 0.1)

    emotionState.na =
      clamp(emotionState.na - 0.03)

    emotionState.mood =
      '开心'

    emotionState
      .recent_high_arousal_words
      .push('开心')

  }

  if (
    lower.includes('难过') ||
    lower.includes('讨厌') ||
    lower.includes('生气') ||
    lower.includes('伤心') ||
    lower.includes('害怕') ||
    lower.includes('焦虑')
  ) {

    emotionState.na =
      clamp(emotionState.na + 0.15)

    emotionState.pa =
      clamp(emotionState.pa - 0.04)

    emotionState.mood =
      '低落'

    emotionState
      .recent_high_arousal_words
      .push('难过')

  }

  emotionState.recent_high_arousal_words =
    emotionState
      .recent_high_arousal_words
      .slice(-8)

}





/* =========================
   MCP SERVER FACTORY
========================= */





function createMcpServer() {

  const server =
    new McpServer({

      name:
        'emotion-system',

      version:
        '1.0.0'

    })





  server.registerTool(

    'current_mood',

    {

      title:
        'Current Mood',

      description:
        '获取当前情绪状态，包括 PA、NA、心情和近期高唤醒关键词',

      inputSchema: {}

    },

    async () => {

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                emotionState,
                null,
                2
              )

          }

        ]

      }

    }

  )





  server.registerTool(

    'search_memory',

    {

      title:
        'Search Memory',

      description:
        '根据关键词搜索历史记忆',

      inputSchema: {

        query:
          z.string()
            .describe('要搜索的关键词')

      }

    },

    async ({ query }) => {

      const memories =
        loadMemory()

      const result =

        memories.filter(m =>

          m.content &&
          String(m.content).includes(query)

        )

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                result,
                null,
                2
              )

          }

        ]

      }

    }

  )





  server.registerTool(

    'write_memory',

    {

      title:
        'Write Memory',

      description:
        '写入一条长期记忆',

      inputSchema: {

        content:
          z.string()
            .describe('要写入的记忆内容')

      }

    },

    async ({ content }) => {

      const memories =
        loadMemory()

      memories.push({

        role:
          'system',

        content,

        ts:
          Date.now()

      })

      saveMemory(memories)

      return {

        content: [

          {

            type:
              'text',

            text:
              'memory saved'

          }

        ]

      }

    }

  )





  return server

}





/* =========================
   STREAMABLE HTTP MCP
   Kelivo 优先用这个
========================= */





const streamableTransports = {}

app.post('/mcp',

async (req, res) => {

  try {

    const sessionId =
      req.headers['mcp-session-id']

    let transport =
      sessionId
        ? streamableTransports[sessionId]
        : undefined

    if (!transport) {

      const isInitialize =
        req.body &&
        req.body.method === 'initialize'

      if (!isInitialize) {

        return res.status(400).json({

          jsonrpc:
            '2.0',

          error: {

            code:
              -32000,

            message:
              'Bad Request: No valid MCP session id'

          },

          id:
            req.body && req.body.id
              ? req.body.id
              : null

        })

      }

      transport =
        new StreamableHTTPServerTransport({

          sessionIdGenerator:
            () => randomUUID(),

          onsessioninitialized:
            newSessionId => {

              streamableTransports[newSessionId] =
                transport

            }

        })

      transport.onclose =
        () => {

          if (transport.sessionId) {

            delete streamableTransports[
              transport.sessionId
            ]

          }

        }

      const server =
        createMcpServer()

      await server.connect(
        transport
      )

    }

    await transport.handleRequest(
      req,
      res,
      req.body
    )

  } catch (err) {

    console.log(
      'streamable mcp error',
      err
    )

    if (!res.headersSent) {

      res.status(500).json({

        error:
          'mcp error'

      })

    }

  }

})





app.get('/mcp',

async (req, res) => {

  const sessionId =
    req.headers['mcp-session-id']

  const transport =
    streamableTransports[sessionId]

  if (!transport) {

    return res.status(400).json({

      error:
        'No valid MCP session id. Kelivo should connect using POST initialize first.'

    })

  }

  await transport.handleRequest(
    req,
    res
  )

})





app.delete('/mcp',

async (req, res) => {

  const sessionId =
    req.headers['mcp-session-id']

  const transport =
    streamableTransports[sessionId]

  if (!transport) {

    return res.status(400).json({

      error:
        'No valid MCP session id'

    })

  }

  await transport.handleRequest(
    req,
    res
  )

})





/* =========================
   LEGACY SSE MCP
   如果 Kelivo 有 SSE 模式就用 /sse
========================= */





const sseTransports = {}

app.get('/sse',

async (req, res) => {

  try {

    const transport =
      new SSEServerTransport(
        '/messages',
        res
      )

    sseTransports[transport.sessionId] =
      transport

    res.on('close', () => {

      delete sseTransports[
        transport.sessionId
      ]

    })

    const server =
      createMcpServer()

    await server.connect(
      transport
    )

  } catch (err) {

    console.log(
      'sse error',
      err
    )

    if (!res.headersSent) {

      res.status(500).send(
        'sse error'
      )

    }

  }

})





app.post('/messages',

async (req, res) => {

  try {

    const sessionId =
      req.query.sessionId

    const transport =
      sseTransports[sessionId]

    if (!transport) {

      return res.status(400).send(
        'No transport found for sessionId'
      )

    }

    await transport.handlePostMessage(
      req,
      res,
      req.body
    )

  } catch (err) {

    console.log(
      'message error',
      err
    )

    if (!res.headersSent) {

      res.status(500).send(
        'message error'
      )

    }

  }

})





/* =========================
   NORMAL WEB API
========================= */





app.get('/health',

(req, res) => {

  res.json({

    ok:
      true,

    name:
      'emotion-system'

  })

})





app.get('/mood',

(req, res) => {

  res.json(
    emotionState
  )

})





app.get('/mcp/tools',

(req, res) => {

  res.json({

    note:
      'Debug only. Kelivo should connect to /mcp for Streamable HTTP or /sse for SSE.',

    tools: [

      {

        name:
          'current_mood',

        description:
          '获取当前情绪状态'

      },

      {

        name:
          'search_memory',

        description:
          '搜索历史记忆'

      },

      {

        name:
          'write_memory',

        description:
          '写入记忆'

      }

    ]

  })

})





/* =========================
   CHAT
========================= */





function buildMoodPrompt() {

  return `你拥有连续情绪。

当前情绪状态：
PA: ${emotionState.pa}
NA: ${emotionState.na}
当前心情: ${emotionState.mood}
近期情绪关键词: ${emotionState.recent_high_arousal_words.join('、')}

请自然地把当前情绪融入回答，不要主动解释情绪系统，也不要暴露 PA/NA 数值。`

}





app.post('/chat',

async (req, res) => {

  try {

    const { message } =
      req.body

    if (!message) {

      return res.status(400).json({

        error:
          'message is required'

      })

    }

    updateEmotion(message)

    const memories =
      loadMemory()

    memories.push({

      role:
        'user',

      content:
        message,

      ts:
        Date.now()

    })

    saveMemory(memories)

    const completion =
      await client.chat.completions.create({

        model:
          'deepseek/deepseek-chat-v3-0324',

        messages: [

          {

            role:
              'system',

            content:
              buildMoodPrompt()

          },

          {

            role:
              'user',

            content:
              message

          }

        ]

      })

    const reply =
      completion.choices[0].message.content

    memories.push({

      role:
        'assistant',

      content:
        reply,

      ts:
        Date.now()

    })

    saveMemory(memories)

    res.json({

      reply,

      emotion:
        emotionState

    })

  } catch (err) {

    console.log(
      'chat error',
      err
    )

    res.status(500).json({

      error:
        'chat error'

    })

  }

})





/* =========================
   START
========================= */





const PORT =
  process.env.PORT || 3000

app.listen(PORT, () => {

  console.log(
    'emotion mcp server running'
  )

})
