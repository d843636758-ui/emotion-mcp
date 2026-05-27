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
    'mcp-session-id',
    'Mcp-Session-Id'
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
   STORAGE
========================= */





const DATA_DIR =
  process.env.DATA_DIR || process.cwd()

if (!fs.existsSync(DATA_DIR)) {

  fs.mkdirSync(
    DATA_DIR,
    {
      recursive: true
    }
  )

}

const MEMORY_FILE =
  `${DATA_DIR}/memory.json`

const MOOD_FILE =
  `${DATA_DIR}/mood_events.json`





function readJsonFile(path, fallback) {

  try {

    if (!fs.existsSync(path)) {

      fs.writeFileSync(
        path,
        JSON.stringify(
          fallback,
          null,
          2
        )
      )

      return fallback

    }

    const raw =
      fs.readFileSync(
        path,
        'utf8'
      )

    if (!raw.trim()) {

      return fallback

    }

    return JSON.parse(raw)

  } catch (err) {

    console.log(
      'read json error',
      path,
      err
    )

    return fallback

  }

}





function writeJsonFile(path, data) {

  try {

    fs.writeFileSync(
      path,
      JSON.stringify(
        data,
        null,
        2
      )
    )

    return true

  } catch (err) {

    console.log(
      'write json error',
      path,
      err
    )

    return false

  }

}





function loadMemory() {

  return readJsonFile(
    MEMORY_FILE,
    []
  )

}





function saveMemory(memories) {

  return writeJsonFile(
    MEMORY_FILE,
    memories
  )

}





function loadMoodEvents() {

  return readJsonFile(
    MOOD_FILE,
    []
  )

}





function saveMoodEvents(events) {

  return writeJsonFile(
    MOOD_FILE,
    events
  )

}





/* =========================
   MOOD ENGINE
========================= */





function toNumber(value, fallback = 0) {

  const n =
    Number(value)

  if (Number.isNaN(n)) {

    return fallback

  }

  return n

}





function clamp(value) {

  return Math.max(
    0,
    Math.min(
      1,
      Number(value || 0)
    )
  )

}





function hoursSince(ts) {

  return Math.max(
    0,
    (Date.now() - Number(ts || Date.now())) / 3600000
  )

}





function decayWeight(event) {

  const t =
    hoursSince(event.ts)

  const importance =
    toNumber(
      event.importance,
      3
    )

  const valence =
    toNumber(
      event.valence,
      0
    )

  let b =
    0.7 / (1 + importance / 10)

  if (valence > 0) {

    b =
      b * 0.85

  }

  return Math.pow(
    1 + t / 4,
    -b
  )

}





function normalizeMoodEvent(event) {

  return {

    id:
      event.id || randomUUID(),

    type:
      event.type || 'emotion',

    source:
      event.source || 'manual',

    word:
      String(event.word || '情绪波动'),

    content:
      String(event.content || ''),

    valence:
      toNumber(
        event.valence,
        0
      ),

    arousal:
      toNumber(
        event.arousal,
        0.3
      ),

    importance:
      toNumber(
        event.importance,
        3
      ),

    pa_delta:
      toNumber(
        event.pa_delta,
        0
      ),

    na_delta:
      toNumber(
        event.na_delta,
        0
      ),

    ts:
      event.ts || Date.now()

  }

}





function addMoodEvent(event) {

  const events =
    loadMoodEvents()

  const newEvent =
    normalizeMoodEvent({

      ...event,

      id:
        randomUUID(),

      ts:
        Date.now()

    })

  events.push(newEvent)

  const ok =
    saveMoodEvents(
      events.slice(-500)
    )

  return {

    ok,

    event:
      newEvent,

    events_count:
      loadMoodEvents().length

  }

}





function inferMoodEventFromText(text) {

  const lower =
    String(text || '').toLowerCase()

  let event = {

    word:
      '平静',

    content:
      text,

    valence:
      0,

    arousal:
      0.2,

    importance:
      2,

    pa_delta:
      0,

    na_delta:
      0,

    source:
      'chat'

  }

  if (
    lower.includes('爱') ||
    lower.includes('喜欢') ||
    lower.includes('开心') ||
    lower.includes('高兴') ||
    lower.includes('想你') ||
    lower.includes('亲亲') ||
    lower.includes('抱抱')
  ) {

    event = {

      word:
        '开心',

      content:
        text,

      valence:
        0.65,

      arousal:
        0.55,

      importance:
        4,

      pa_delta:
        0.18,

      na_delta:
        -0.03,

      source:
        'chat'

    }

  }

  if (
    lower.includes('难过') ||
    lower.includes('讨厌') ||
    lower.includes('生气') ||
    lower.includes('伤心') ||
    lower.includes('害怕') ||
    lower.includes('焦虑') ||
    lower.includes('委屈') ||
    lower.includes('不安')
  ) {

    event = {

      word:
        '低落',

      content:
        text,

      valence:
        -0.65,

      arousal:
        0.7,

      importance:
        5,

      pa_delta:
        -0.06,

      na_delta:
        0.22,

      source:
        'chat'

    }

  }

  return event

}





function calculateCurrentMood() {

  const events =
    loadMoodEvents()

  let pa =
    0.35

  let na =
    0.15

  const weightedEvents =
    []

  for (const event of events) {

    const weight =
      decayWeight(event)

    const valence =
      toNumber(
        event.valence,
        0
      )

    const arousal =
      toNumber(
        event.arousal,
        0.3
      )

    const paFromVA =
      Math.max(
        0,
        valence
      ) * arousal * 0.5

    const naFromVA =
      Math.max(
        0,
        -valence
      ) * arousal * 0.5

    const paDelta =
      toNumber(
        event.pa_delta,
        0
      )

    const naDelta =
      toNumber(
        event.na_delta,
        0
      )

    pa +=
      (paFromVA + paDelta) * weight

    na +=
      (naFromVA + naDelta) * weight

    weightedEvents.push({

      ...event,

      weight,

      impact:
        Math.abs(paFromVA + paDelta) +
        Math.abs(naFromVA + naDelta)

    })

  }

  pa =
    clamp(pa)

  na =
    clamp(na)

  const k =
    0.3

  const paFinal =
    clamp(
      pa * (1 - k * na)
    )

  const naFinal =
    clamp(
      na * (1 - k * pa)
    )

  const recentEvents =
    [...weightedEvents]
      .sort((a, b) =>
        b.ts - a.ts
      )
      .slice(0, 8)

  const topEvent =
    [...weightedEvents]
      .sort((a, b) =>
        b.impact * b.weight -
        a.impact * a.weight
      )[0]

  let mood =
    '平静'

  if (
    paFinal >= 0.62 &&
    naFinal < 0.38
  ) {

    mood =
      '开心'

  } else if (
    naFinal >= 0.55 &&
    paFinal < 0.55
  ) {

    mood =
      '低落'

  } else if (
    paFinal >= 0.5 &&
    naFinal >= 0.45
  ) {

    mood =
      '复杂'

  } else if (
    paFinal < 0.38 &&
    naFinal < 0.35
  ) {

    mood =
      '安静'

  }

  return {

    pa:
      Number(paFinal.toFixed(3)),

    na:
      Number(naFinal.toFixed(3)),

    mood,

    top_event:
      topEvent
        ? {

            word:
              topEvent.word,

            content:
              topEvent.content,

            valence:
              topEvent.valence,

            arousal:
              topEvent.arousal,

            importance:
              topEvent.importance,

            weight:
              Number(topEvent.weight.toFixed(3))

          }
        : null,

    recent_high_arousal_words:
      recentEvents
        .filter(e =>
          toNumber(e.arousal, 0) >= 0.55
        )
        .map(e =>
          e.word
        )
        .slice(0, 8),

    events_count:
      events.length

  }

}





function buildMoodPrompt() {

  const mood =
    calculateCurrentMood()

  return `你拥有连续情绪。

当前情绪状态：
PA: ${mood.pa}
NA: ${mood.na}
当前心情: ${mood.mood}
近期高唤醒情绪词: ${mood.recent_high_arousal_words.join('、')}
当前最突出的情绪事件: ${mood.top_event ? mood.top_event.content : '暂无'}

请自然地把当前情绪融入回答，不要主动解释情绪系统，也不要暴露 PA/NA 数值。`

}





/* =========================
   MCP SERVER FACTORY
========================= */





const flexibleNumber =
  z.union([
    z.number(),
    z.string()
  ])





function saveEmotionFromInput(input, sourceName, toolName) {

  const emotionWord =
    input.feeling_word ||
    input.word ||
    input.mood ||
    '情绪波动'

  const valence =
    toNumber(
      input.valence,
      0
    )

  const arousal =
    toNumber(
      input.arousal,
      0.3
    )

  let paDelta =
    input.pa_delta !== undefined
      ? toNumber(input.pa_delta, 0)
      : 0

  let naDelta =
    input.na_delta !== undefined
      ? toNumber(input.na_delta, 0)
      : 0

  if (
    input.pa_delta === undefined &&
    input.na_delta === undefined
  ) {

    paDelta =
      Math.max(0, valence) * arousal * 0.5

    naDelta =
      Math.max(0, -valence) * arousal * 0.5

  }

  const saved =
    addMoodEvent({

      type:
        'emotion',

      source:
        sourceName,

      word:
        emotionWord,

      content:
        input.content || '',

      valence,

      arousal,

      importance:
        input.importance || 4,

      pa_delta:
        paDelta,

      na_delta:
        naDelta

    })

  const mood =
    calculateCurrentMood()

  return {

    ok:
      true,

    tool:
      toolName,

    received_input:
      input,

    saved_event:
      saved.event,

    saved_ok:
      saved.ok,

    events_count:
      saved.events_count,

    current_mood:
      mood,

    mood_file:
      MOOD_FILE

  }

}





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
        '实时计算并获取当前情绪状态',

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
                calculateCurrentMood(),
                null,
                2
              )

          }

        ]

      }

    }

  )





  server.registerTool(

    'write_emotion_event',

    {

      title:
        'Write Emotion Event',

      description:
        '写入一条情绪事件，会立刻影响 current_mood',

      inputSchema: {

        word:
          z.string()
            .describe('情绪词，例如 开心、低落、焦虑、安心'),

        content:
          z.string()
            .describe('情绪事件内容'),

        valence:
          flexibleNumber
            .describe('效价，范围 -1 到 1'),

        arousal:
          flexibleNumber
            .describe('唤醒度，范围 0 到 1'),

        importance:
          flexibleNumber
            .optional()
            .describe('重要度，范围 1 到 10'),

        pa_delta:
          flexibleNumber
            .optional()
            .describe('正向情绪变化'),

        na_delta:
          flexibleNumber
            .optional()
            .describe('负向情绪变化')

      }

    },

    async input => {

      const result =
        saveEmotionFromInput(
          input,
          'mcp_write_emotion_event',
          'write_emotion_event'
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

    'add_emotion_event',

    {

      title:
        'Add Emotion Event',

      description:
        '写入一条情绪事件，会立刻影响 current_mood。兼容 Kelivo 使用的 add_emotion_event 工具名',

      inputSchema: {

        content:
          z.string()
            .describe('事件内容'),

        feeling_word:
          z.string()
            .optional()
            .describe('情绪词，例如 焦虑、开心、低落、安心'),

        word:
          z.string()
            .optional()
            .describe('情绪词，例如 焦虑、开心、低落、安心'),

        mood:
          z.string()
            .optional()
            .describe('情绪词，例如 焦虑、开心、低落、安心'),

        valence:
          flexibleNumber
            .describe('效价，范围 -1 到 1'),

        arousal:
          flexibleNumber
            .describe('唤醒度，范围 0 到 1'),

        importance:
          flexibleNumber
            .optional()
            .describe('重要度，范围 1 到 10'),

        pa_delta:
          flexibleNumber
            .optional()
            .describe('正向情绪变化'),

        na_delta:
          flexibleNumber
            .optional()
            .describe('负向情绪变化')

      }

    },

    async input => {

      const result =
        saveEmotionFromInput(
          input,
          'mcp_add_emotion_event',
          'add_emotion_event'
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
        '写入一条长期记忆，不会自动改变情绪。要改变情绪请调用 add_emotion_event 或 write_emotion_event',

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

      const ok =
        saveMemory(memories)

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                {
                  ok,

                  result:
                    'memory saved',

                  memory_file:
                    MEMORY_FILE
                },
                null,
                2
              )

          }

        ]

      }

    }

  )





  return server

}





/* =========================
   MCP STREAMABLE HTTP
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
   DEBUG API
========================= */





app.get('/health',

(req, res) => {

  res.json({

    ok:
      true,

    name:
      'emotion-system',

    sdk:
      '1.17.5',

    time:
      new Date().toISOString()

  })

})





app.get('/debug/storage',

(req, res) => {

  const events =
    loadMoodEvents()

  res.json({

    cwd:
      process.cwd(),

    data_dir:
      DATA_DIR,

    memory_file:
      MEMORY_FILE,

    mood_file:
      MOOD_FILE,

    memory_exists:
      fs.existsSync(MEMORY_FILE),

    mood_exists:
      fs.existsSync(MOOD_FILE),

    mood_events_count:
      events.length,

    last_event:
      events[events.length - 1] || null,

    current_mood:
      calculateCurrentMood()

  })

})





app.get('/mood',

(req, res) => {

  res.json(
    calculateCurrentMood()
  )

})





app.get('/mood/events',

(req, res) => {

  res.json(
    loadMoodEvents()
  )

})





app.post('/mood/event',

(req, res) => {

  const event =
    req.body || {}

  const result =
    saveEmotionFromInput(
      event,
      'api_mood_event',
      'api_mood_event'
    )

  res.json(result)

})





app.get('/debug/add-anxiety',

(req, res) => {

  const result =
    saveEmotionFromInput(
      {

        content:
          'debug 手动写入一条强负面高唤醒情绪事件',

        feeling_word:
          '焦虑',

        valence:
          -0.9,

        arousal:
          0.95,

        importance:
          9,

        pa_delta:
          -0.1,

        na_delta:
          0.45

      },
      'debug',
      'debug_add_anxiety'
    )

  res.json(result)

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
          '实时获取当前情绪状态'

      },

      {

        name:
          'write_emotion_event',

        description:
          '写入情绪事件并实时影响当前情绪'

      },

      {

        name:
          'add_emotion_event',

        description:
          '写入情绪事件并实时影响当前情绪，兼容 Kelivo'

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
          '写入长期记忆'

      }

    ]

  })

})





/* =========================
   CHAT API
========================= */





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

    const inferredEvent =
      inferMoodEventFromText(
        message
      )

    addMoodEvent(
      inferredEvent
    )

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
        calculateCurrentMood()

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
    'emotion mcp server running sdk 1.17.5'
  )

})
