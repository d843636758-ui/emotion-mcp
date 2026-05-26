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
    'mcp-protocol-version',
    'accept'
  ]
}))

app.use(express.json())

app.use(express.static('public'))





/* =========================
   OPENAI
========================= */





const client = new OpenAI({

  apiKey:
    process.env.OPENAI_API_KEY || 'missing-key',

  baseURL:
    process.env.OPENAI_BASE_URL ||
    'https://openrouter.ai/api/v1'

})





/* =========================
   FILE STORAGE
========================= */





const EVENT_FILE =
  './emotion-events.json'

const STATE_FILE =
  './emotion-state.json'

const DECORATION_FILE =
  './decoration-moods.json'





function readJson(file, fallback) {

  try {

    if (!fs.existsSync(file)) {

      fs.writeFileSync(
        file,
        JSON.stringify(fallback, null, 2)
      )

      return fallback

    }

    return JSON.parse(
      fs.readFileSync(file, 'utf8')
    )

  } catch {

    return fallback

  }

}





function writeJson(file, data) {

  try {

    fs.writeFileSync(
      file,
      JSON.stringify(data, null, 2)
    )

  } catch (err) {

    console.log(
      'write json error',
      file,
      err
    )

  }

}





/* =========================
   DEFAULT DATA
========================= */





function defaultState() {

  return {

    pa: 0.45,

    na: 0.12,

    mood: '平静',

    affection_level: 70,

    recent_high_arousal_words: []

  }

}





function defaultDecorations() {

  return [

    {

      id:
        'deco_1',

      word:
        '整理完房间',

      feeling_word:
        '安心',

      valence:
        0.45,

      arousal:
        0.28,

      importance:
        1,

      kind:
        'daily'

    },

    {

      id:
        'deco_2',

      word:
        '窗外下雨',

      feeling_word:
        '安静',

      valence:
        0.12,

      arousal:
        0.18,

      importance:
        1,

      kind:
        'ambient'

    },

    {

      id:
        'deco_3',

      word:
        '想到一件很在意的事',

      feeling_word:
        '牵挂',

      valence:
        -0.12,

      arousal:
        0.52,

      importance:
        2,

      kind:
        'memory'

    },

    {

      id:
        'deco_4',

      word:
        '刚刚被温柔对待',

      feeling_word:
        '心软',

      valence:
        0.56,

      arousal:
        0.48,

      importance:
        2,

      kind:
        'sparkle'

    }

  ]

}





/* =========================
   HELPERS
========================= */





function now() {

  return Date.now()

}





function clamp(value) {

  return Math.max(
    0,
    Math.min(1, Number(value) || 0)
  )

}





function clampValence(value) {

  return Math.max(
    -1,
    Math.min(1, Number(value) || 0)
  )

}





function hoursSince(ts) {

  return Math.max(
    0,
    (now() - Number(ts || now())) / 3600000
  )

}





function todayKey() {

  const date =
    new Date()

  return [
    date.getUTCFullYear(),
    date.getUTCMonth() + 1,
    date.getUTCDate()
  ].join('-')

}





function hashString(input) {

  let hash =
    0

  for (let i = 0; i < input.length; i++) {

    hash =
      ((hash << 5) - hash) +
      input.charCodeAt(i)

    hash |= 0

  }

  return Math.abs(hash)

}





function getTodayDecoration() {

  const decorations =
    readJson(
      DECORATION_FILE,
      defaultDecorations()
    )

  if (!decorations.length) {

    return null

  }

  const index =
    hashString(todayKey()) % decorations.length

  return decorations[index]

}





/* =========================
   POWER LAW DECAY
========================= */





function powerLawWeight(event) {

  const t =
    hoursSince(event.ts)

  const importance =
    Math.max(
      1,
      Math.min(10, Number(event.importance || 3))
    )

  let bEff =
    0.7 / (1 + importance / 10)

  const valence =
    Number(event.valence || 0)

  if (valence > 0) {

    bEff *= 0.85

  }

  const tau =
    4

  return Math.pow(
    1 + t / tau,
    -bEff
  )

}





function eventToPANAS(event) {

  const v =
    clampValence(event.valence)

  const a =
    clamp(event.arousal)

  const paDelta =
    typeof event.pa_delta === 'number'
      ? event.pa_delta
      : Math.max(0, v) * a * 0.5

  const naDelta =
    typeof event.na_delta === 'number'
      ? event.na_delta
      : Math.max(0, -v) * a * 0.5

  return {

    pa_delta:
      paDelta,

    na_delta:
      naDelta

  }

}





function applyESM(pa, na) {

  const k =
    0.3

  const rawPa =
    clamp(pa)

  const rawNa =
    clamp(na)

  return {

    pa:
      clamp(rawPa * (1 - k * rawNa)),

    na:
      clamp(rawNa * (1 - k * rawPa))

  }

}





function labelMood(pa, na, fallback) {

  if (pa >= 0.65 && na < 0.35) {

    return '开心'

  }

  if (na >= 0.65 && pa < 0.35) {

    return '低落'

  }

  if (pa >= 0.45 && na >= 0.45) {

    return '复杂'

  }

  if (na >= 0.4) {

    return '不安'

  }

  if (pa >= 0.45) {

    return '安心'

  }

  return fallback || '平静'

}





function buildSnapshot() {

  const state =
    readJson(
      STATE_FILE,
      defaultState()
    )

  const events =
    readJson(
      EVENT_FILE,
      []
    )

  const decoration =
    getTodayDecoration()

  const allEvents =
    [...events]

  if (decoration) {

    allEvents.push({

      id:
        'today_decoration',

      source:
        'decoration',

      content:
        decoration.word,

      feeling_word:
        decoration.feeling_word,

      valence:
        decoration.valence,

      arousal:
        decoration.arousal,

      importance:
        decoration.importance || 1,

      ts:
        now(),

      decoration:
        true

    })

  }

  let pa =
    Number(state.pa || 0.45)

  let na =
    Number(state.na || 0.12)

  const weightedEvents =
    []

  for (const event of allEvents) {

    const weight =
      powerLawWeight(event)

    const panas =
      eventToPANAS(event)

    pa +=
      panas.pa_delta * weight

    na +=
      panas.na_delta * weight

    weightedEvents.push({

      ...event,

      weight,

      weighted_pa:
        panas.pa_delta * weight,

      weighted_na:
        panas.na_delta * weight

    })

  }

  const esm =
    applyESM(pa, na)

  const recentHighArousal =
    weightedEvents
      .filter(e => Number(e.arousal || 0) >= 0.55)
      .sort((a, b) => Number(b.ts) - Number(a.ts))
      .slice(0, 8)
      .map(e =>
        e.feeling_word ||
        e.word ||
        e.content ||
        '情绪'
      )

  const surfaced =
    weightedEvents
      .sort((a, b) => {

        const scoreA =
          Number(a.importance || 1) *
          Number(a.weight || 0)

        const scoreB =
          Number(b.importance || 1) *
          Number(b.weight || 0)

        return scoreB - scoreA

      })[0] || null

  const mood =
    labelMood(
      esm.pa,
      esm.na,
      state.mood
    )

  const affection =
    clamp(
      Number(state.affection_level || 70) / 100 +
      esm.pa * 0.08 -
      esm.na * 0.05
    ) * 100

  return {

    ok:
      true,

    pa:
      Number(esm.pa.toFixed(4)),

    na:
      Number(esm.na.toFixed(4)),

    mood,

    decoration_mood:
      decoration,

    surfaced_event_top1:
      surfaced
        ? {
            content:
              surfaced.content,
            feeling_word:
              surfaced.feeling_word,
            source:
              surfaced.source,
            weight:
              Number(surfaced.weight.toFixed(4))
          }
        : null,

    affection_level:
      Number(affection.toFixed(2)),

    recent_high_arousal_words:
      recentHighArousal,

    event_count:
      events.length,

    server_time:
      new Date().toISOString()

  }

}





function buildMoodPromptFromSnapshot(snapshot) {

  const parts =
    []

  if (snapshot.decoration_mood) {

    parts.push(
      `此刻的日常状态：${snapshot.decoration_mood.word}，对应情绪是${snapshot.decoration_mood.feeling_word}`
    )

  }

  if (snapshot.pa > 0.55 && snapshot.na < 0.35) {

    parts.push(
      '最近正向情感较高，整体更柔和、明亮、愿意靠近'
    )

  }

  if (snapshot.na > 0.45) {

    parts.push(
      '最近负向情感偏高，心里有些不安、紧绷或低落'
    )

  }

  if (snapshot.surfaced_event_top1) {

    parts.push(
      `脑子里最挂念的事：${snapshot.surfaced_event_top1.content}`
    )

  }

  if (
    snapshot.recent_high_arousal_words &&
    snapshot.recent_high_arousal_words.length
  ) {

    parts.push(
      `近期高唤醒情绪关键词：${snapshot.recent_high_arousal_words.join('、')}`
    )

  }

  parts.push(
    '预设心情只是背景色，用户当前这条消息是前景，前景优先'
  )

  return `【当下心情状态】
${parts.join('\n')}

请自然融入回答，不要主动点破这些状态，不要暴露 PA、NA 数值。`

}





/* =========================
   MUTATIONS
========================= */





function updateBaseMood(input) {

  const state =
    readJson(
      STATE_FILE,
      defaultState()
    )

  if (typeof input.pa === 'number') {

    state.pa =
      clamp(input.pa)

  }

  if (typeof input.na === 'number') {

    state.na =
      clamp(input.na)

  }

  if (typeof input.mood === 'string') {

    state.mood =
      input.mood

  }

  if (typeof input.affection_level === 'number') {

    state.affection_level =
      Math.max(
        0,
        Math.min(100, input.affection_level)
      )

  }

  if (typeof input.keyword === 'string') {

    state
      .recent_high_arousal_words
      .push(input.keyword)

    state.recent_high_arousal_words =
      state
        .recent_high_arousal_words
        .slice(-8)

  }

  writeJson(
    STATE_FILE,
    state
  )

  return state

}





function addEmotionEvent(input) {

  const events =
    readJson(
      EVENT_FILE,
      []
    )

  const event = {

    id:
      randomUUID(),

    source:
      input.source || 'conversation',

    content:
      input.content || '',

    feeling_word:
      input.feeling_word || input.mood || '情绪',

    valence:
      clampValence(input.valence),

    arousal:
      clamp(input.arousal),

    pa_delta:
      typeof input.pa_delta === 'number'
        ? input.pa_delta
        : undefined,

    na_delta:
      typeof input.na_delta === 'number'
        ? input.na_delta
        : undefined,

    importance:
      Math.max(
        1,
        Math.min(10, Number(input.importance || 3))
      ),

    resolved:
      input.resolved ? 1 : 0,

    ts:
      now()

  }

  events.push(event)

  writeJson(
    EVENT_FILE,
    events
  )

  return event

}





function runDecay() {

  const events =
    readJson(
      EVENT_FILE,
      []
    )

  const enriched =
    events.map(event => {

      const weight =
        powerLawWeight(event)

      return {

        ...event,

        decay_score:
          Number(weight.toFixed(6))

      }

    })

  writeJson(
    EVENT_FILE,
    enriched
  )

  return enriched

}





function scanTextEmotion(text) {

  const source =
    String(text || '')

  const lexicon = [

    {
      word:
        '爱',
      v:
        0.8,
      a:
        0.65
    },

    {
      word:
        '喜欢',
      v:
        0.65,
      a:
        0.45
    },

    {
      word:
        '开心',
      v:
        0.7,
      a:
        0.55
    },

    {
      word:
        '想你',
      v:
        0.55,
      a:
        0.7
    },

    {
      word:
        '难过',
      v:
        -0.65,
      a:
        0.55
    },

    {
      word:
        '讨厌',
      v:
        -0.8,
      a:
        0.75
    },

    {
      word:
        '生气',
      v:
        -0.7,
      a:
        0.8
    },

    {
      word:
        '害怕',
      v:
        -0.72,
      a:
        0.82
    },

    {
      word:
        '焦虑',
      v:
        -0.6,
      a:
        0.78
    }

  ]

  const hits =
    lexicon.filter(item =>
      source.includes(item.word)
    )

  if (!hits.length) {

    return {

      hits: [],

      avgV: 0,

      minV: 0,

      maxV: 0,

      intensity: 0,

      should_score_now: false

    }

  }

  const avgV =
    hits.reduce((sum, item) => sum + item.v, 0) /
    hits.length

  const minV =
    Math.min(...hits.map(item => item.v))

  const maxV =
    Math.max(...hits.map(item => item.v))

  const intensity =
    Math.min(
      1,
      hits.reduce((sum, item) => sum + item.a, 0) /
      hits.length
    )

  const shouldScoreNow =
    minV < -0.5 ||
    maxV > 0.7 ||
    intensity >= 0.7

  return {

    hits,

    avgV,

    minV,

    maxV,

    intensity,

    should_score_now:
      shouldScoreNow

  }

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
        '2.0.0'

    })





  server.registerTool(

    'current_mood',

    {

      title:
        'Current Mood',

      description:
        '读取当前心情快照，包括 PA、NA、装饰心情、浮现事件、好感度和高唤醒关键词',

      inputSchema: {}

    },

    async () => {

      const snapshot =
        buildSnapshot()

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                snapshot,
                null,
                2
              )

          }

        ]

      }

    }

  )





  server.registerTool(

    'update_mood',

    {

      title:
        'Update Mood',

      description:
        '直接更新当前基础心情状态',

      inputSchema: {

        pa:
          z.number()
            .min(0)
            .max(1)
            .optional()
            .describe('正向情感，0 到 1'),

        na:
          z.number()
            .min(0)
            .max(1)
            .optional()
            .describe('负向情感，0 到 1'),

        mood:
          z.string()
            .optional()
            .describe('当前心情文字'),

        affection_level:
          z.number()
            .min(0)
            .max(100)
            .optional()
            .describe('好感度，0 到 100'),

        keyword:
          z.string()
            .optional()
            .describe('近期情绪关键词')

      }

    },

    async input => {

      const state =
        updateBaseMood(input)

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                state,
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
        '写入一条情绪事件，进入情绪事件流并参与衰减和 current_mood 计算',

      inputSchema: {

        content:
          z.string()
            .describe('事件内容'),

        feeling_word:
          z.string()
            .optional()
            .describe('情绪词，例如开心、焦虑、心疼'),

        valence:
          z.number()
            .min(-1)
            .max(1)
            .describe('效价，-1 极负面，1 极正面'),

        arousal:
          z.number()
            .min(0)
            .max(1)
            .describe('唤醒度，0 平静，1 强烈'),

        importance:
          z.number()
            .min(1)
            .max(10)
            .optional()
            .describe('重要度，1 到 10'),

        pa_delta:
          z.number()
            .optional()
            .describe('正向情感增量，可省略'),

        na_delta:
          z.number()
            .optional()
            .describe('负向情感增量，可省略'),

        source:
          z.string()
            .optional()
            .describe('事件来源，例如 conversation、manual、decoration'),

        resolved:
          z.boolean()
            .optional()
            .describe('是否已解决')

      }

    },

    async input => {

      const event =
        addEmotionEvent(input)

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                event,
                null,
                2
              )

          }

        ]

      }

    }

  )





  server.registerTool(

    'list_emotion_events',

    {

      title:
        'List Emotion Events',

      description:
        '查看情绪事件流，默认返回最近 20 条',

      inputSchema: {

        limit:
          z.number()
            .min(1)
            .max(100)
            .optional()
            .describe('返回数量')

      }

    },

    async ({ limit }) => {

      const events =
        readJson(
          EVENT_FILE,
          []
        )

      const result =
        events
          .slice()
          .sort((a, b) => b.ts - a.ts)
          .slice(0, limit || 20)

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

    'run_emotion_decay',

    {

      title:
        'Run Emotion Decay',

      description:
        '执行一次幂律衰减计算，为情绪事件写入 decay_score',

      inputSchema: {}

    },

    async () => {

      const events =
        runDecay()

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                {
                  ok: true,
                  count: events.length
                },
                null,
                2
              )

          }

        ]

      }

    }

  )





  server.registerTool(

    'get_mood_prompt',

    {

      title:
        'Get Mood Prompt',

      description:
        '生成可注入模型 system prompt 的当下心情提示词',

      inputSchema: {}

    },

    async () => {

      const snapshot =
        buildSnapshot()

      const prompt =
        buildMoodPromptFromSnapshot(snapshot)

      return {

        content: [

          {

            type:
              'text',

            text:
              prompt

          }

        ]

      }

    }

  )





  server.registerTool(

    'add_decoration_mood',

    {

      title:
        'Add Decoration Mood',

      description:
        '新增一条装饰心情，用于模拟角色自己的日常状态',

      inputSchema: {

        word:
          z.string()
            .describe('日常事件描述，例如整理唱片、刚跑完步'),

        feeling_word:
          z.string()
            .describe('对应情绪词'),

        valence:
          z.number()
            .min(-1)
            .max(1)
            .describe('效价'),

        arousal:
          z.number()
            .min(0)
            .max(1)
            .describe('唤醒度'),

        importance:
          z.number()
            .min(1)
            .max(10)
            .optional()
            .describe('重要度'),

        kind:
          z.string()
            .optional()
            .describe('类型，例如 daily、sparkle、ambient')

      }

    },

    async input => {

      const decorations =
        readJson(
          DECORATION_FILE,
          defaultDecorations()
        )

      const item = {

        id:
          randomUUID(),

        word:
          input.word,

        feeling_word:
          input.feeling_word,

        valence:
          clampValence(input.valence),

        arousal:
          clamp(input.arousal),

        importance:
          input.importance || 1,

        kind:
          input.kind || 'daily'

      }

      decorations.push(item)

      writeJson(
        DECORATION_FILE,
        decorations
      )

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                item,
                null,
                2
              )

          }

        ]

      }

    }

  )





  server.registerTool(

    'list_decoration_moods',

    {

      title:
        'List Decoration Moods',

      description:
        '列出装饰心情池',

      inputSchema: {}

    },

    async () => {

      const decorations =
        readJson(
          DECORATION_FILE,
          defaultDecorations()
        )

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                decorations,
                null,
                2
              )

          }

        ]

      }

    }

  )





  server.registerTool(

    'scan_text_emotion',

    {

      title:
        'Scan Text Emotion',

      description:
        '本地情绪漏斗，扫描文本情绪强度，判断是否需要立即写入情绪事件',

      inputSchema: {

        text:
          z.string()
            .describe('要扫描的文本')

      }

    },

    async ({ text }) => {

      const result =
        scanTextEmotion(text)

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





  return server

}





/* =========================
   STREAMABLE HTTP MCP
========================= */





const streamableTransports =
  {}

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
        'No valid MCP session id. Use POST initialize first.'

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





const sseTransports =
  {}

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
   WEB API
========================= */





app.get('/health',

(req, res) => {

  res.json({

    ok:
      true,

    name:
      'emotion-system',

    version:
      '2.0.0',

    tools: [

      'current_mood',

      'update_mood',

      'add_emotion_event',

      'list_emotion_events',

      'run_emotion_decay',

      'get_mood_prompt',

      'add_decoration_mood',

      'list_decoration_moods',

      'scan_text_emotion'

    ]

  })

})





app.get('/mood',

(req, res) => {

  res.json(
    buildSnapshot()
  )

})





app.get('/mcp/tools',

(req, res) => {

  res.json({

    note:
      'Debug only. Kelivo should connect to /mcp for Streamable HTTP or /sse for SSE.',

    tools: [

      'current_mood',

      'update_mood',

      'add_emotion_event',

      'list_emotion_events',

      'run_emotion_decay',

      'get_mood_prompt',

      'add_decoration_mood',

      'list_decoration_moods',

      'scan_text_emotion'

    ]

  })

})





app.post('/emotion/event',

(req, res) => {

  const event =
    addEmotionEvent(req.body || {})

  res.json(event)

})





app.post('/emotion/update',

(req, res) => {

  const state =
    updateBaseMood(req.body || {})

  res.json(state)

})





/* =========================
   CHAT DEMO
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

    const funnel =
      scanTextEmotion(message)

    if (
      funnel.should_score_now &&
      funnel.hits.length
    ) {

      addEmotionEvent({

        content:
          message,

        feeling_word:
          funnel.hits[0].word,

        valence:
          funnel.avgV,

        arousal:
          funnel.intensity,

        importance:
          4,

        source:
          'conversation'

      })

    }

    const snapshot =
      buildSnapshot()

    const prompt =
      buildMoodPromptFromSnapshot(snapshot)

    const completion =
      await client.chat.completions.create({

        model:
          process.env.CHAT_MODEL ||
          'deepseek/deepseek-chat-v3-0324',

        messages: [

          {

            role:
              'system',

            content:
              prompt

          },

          {

            role:
              'user',

            content:
              message

          }

        ]

      })

    res.json({

      reply:
        completion.choices[0].message.content,

      emotion:
        buildSnapshot()

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
