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

app.use(express.json({
  limit: '20mb'
}))

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

const PERSONALITY_FILE =
  `${DATA_DIR}/personality_state.json`

const RELATIONSHIP_FILE =
  `${DATA_DIR}/relationship_state.json`





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





function defaultPersonalityState() {

  return {

    profile:
      'warm_sensitive',

    baseline_pa:
      0.38,

    baseline_na:
      0.18,

    sensitivity:
      0.72,

    attachment_sensitivity:
      0.78,

    inertia_strength:
      0.28,

    repair_need_strength:
      0.75,

    positive_repair_power:
      0.55,

    negative_memory_bias:
      0.62,

    description:
      '温柔、敏感、重视关系安全感。正向安抚会明显修复情绪，但高唤醒负面事件会留下未完成感。'

  }

}





function defaultRelationshipState() {

  return {

    safety:
      0.68,

    trust:
      0.7,

    closeness:
      0.72,

    relationship_profile:
      'close_and_sensitive',

    description:
      '亲近、信任较高，但关系安全感仍会被冷淡、误解、争执和离开感影响。',

    unresolved_count_cache:
      0,

    last_mood_snapshot:
      null,

    last_updated:
      Date.now()

  }

}





function loadPersonalityState() {

  return readJsonFile(
    PERSONALITY_FILE,
    defaultPersonalityState()
  )

}





function savePersonalityState(state) {

  return writeJsonFile(
    PERSONALITY_FILE,
    state
  )

}





function loadRelationshipState() {

  return readJsonFile(
    RELATIONSHIP_FILE,
    defaultRelationshipState()
  )

}





function saveRelationshipState(state) {

  return writeJsonFile(
    RELATIONSHIP_FILE,
    state
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

  const personality =
    loadPersonalityState()

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

  const unresolved =
    event.resolved === true
      ? false
      : Boolean(event.need_repair)

  let b =
    0.7 / (1 + importance / 10)

  if (valence > 0) {

    b =
      b * 0.85

  }

  if (unresolved && valence < 0) {

    b =
      b * 0.72

  }

  let weight =
    Math.pow(
      1 + t / 4,
      -b
    )

  if (event.resolved === true) {

    weight =
      weight * 0.35

  }

  if (unresolved && valence < 0) {

    weight =
      weight *
      (1 + personality.repair_need_strength * 0.55)

  }

  return weight

}





function inferNeedRepair(event) {

  const valence =
    toNumber(
      event.valence,
      0
    )

  const arousal =
    toNumber(
      event.arousal,
      0
    )

  const importance =
    toNumber(
      event.importance,
      3
    )

  const text =
    `${event.word || ''} ${event.content || ''}`

  const hasRelationalPain =
    text.includes('不要') ||
    text.includes('讨厌') ||
    text.includes('冷淡') ||
    text.includes('离开') ||
    text.includes('害怕') ||
    text.includes('焦虑') ||
    text.includes('委屈') ||
    text.includes('难过') ||
    text.includes('生气')

  return (
    valence < -0.35 &&
    arousal >= 0.45 &&
    importance >= 4
  ) || hasRelationalPain

}





function normalizeMoodEvent(event) {

  const normalized = {

    id:
      event.id || randomUUID(),

    type:
      event.type || 'emotion',

    source:
      event.source || 'manual',

    word:
      String(
        event.word ||
        event.feeling_word ||
        event.mood ||
        '情绪波动'
      ),

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

    safety_delta:
      toNumber(
        event.safety_delta,
        0
      ),

    need_repair:
      event.need_repair === undefined
        ? false
        : Boolean(event.need_repair),

    resolved:
      Boolean(event.resolved),

    resolved_at:
      event.resolved_at || null,

    repair_note:
      event.repair_note || null,

    ts:
      event.ts || Date.now()

  }

  if (event.need_repair === undefined) {

    normalized.need_repair =
      inferNeedRepair(normalized)

  }

  return normalized

}





function updateRelationshipFromEvent(event) {

  const relationship =
    loadRelationshipState()

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

  const importance =
    toNumber(
      event.importance,
      3
    )

  const explicitSafetyDelta =
    toNumber(
      event.safety_delta,
      0
    )

  const autoSafetyDelta =
    valence * arousal * importance * 0.018

  relationship.safety =
    clamp(
      relationship.safety +
      explicitSafetyDelta +
      autoSafetyDelta
    )

  if (valence > 0.25) {

    relationship.trust =
      clamp(
        relationship.trust +
        valence * arousal * 0.035
      )

    relationship.closeness =
      clamp(
        relationship.closeness +
        valence * arousal * 0.03
      )

  }

  if (valence < -0.25) {

    relationship.trust =
      clamp(
        relationship.trust +
        valence * arousal * 0.025
      )

    relationship.safety =
      clamp(
        relationship.safety +
        valence * arousal * 0.035
      )

  }

  relationship.last_updated =
    Date.now()

  saveRelationshipState(
    relationship
  )

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
      events.slice(-800)
    )

  updateRelationshipFromEvent(
    newEvent
  )

  const mood =
    calculateCurrentMood({
      useInertia:
        true
    })

  const relationship =
    loadRelationshipState()

  relationship.last_mood_snapshot = {

    pa:
      mood.pa,

    na:
      mood.na,

    mood:
      mood.mood,

    ts:
      Date.now()

  }

  relationship.unresolved_count_cache =
    mood.unresolved_events_count

  relationship.last_updated =
    Date.now()

  saveRelationshipState(
    relationship
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
      'chat',

    safety_delta:
      0

  }

  if (
    lower.includes('爱') ||
    lower.includes('喜欢') ||
    lower.includes('开心') ||
    lower.includes('高兴') ||
    lower.includes('想你') ||
    lower.includes('亲亲') ||
    lower.includes('抱抱') ||
    lower.includes('安心') ||
    lower.includes('谢谢')
  ) {

    event = {

      word:
        '被安抚',

      content:
        text,

      valence:
        0.68,

      arousal:
        0.5,

      importance:
        4,

      pa_delta:
        0.16,

      na_delta:
        -0.06,

      safety_delta:
        0.06,

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
    lower.includes('不安') ||
    lower.includes('不要你') ||
    lower.includes('离开')
  ) {

    event = {

      word:
        '受伤',

      content:
        text,

      valence:
        -0.68,

      arousal:
        0.72,

      importance:
        6,

      pa_delta:
        -0.07,

      na_delta:
        0.24,

      safety_delta:
        -0.08,

      source:
        'chat',

      need_repair:
        true

    }

  }

  return event

}





function getUnresolvedEvents(events) {

  return events
    .filter(event =>
      event.need_repair &&
      event.resolved !== true
    )
    .sort((a, b) =>
      b.ts - a.ts
    )

}





function calculateCurrentMood(options = {}) {

  const useInertia =
    options.useInertia !== false

  const events =
    loadMoodEvents()

  const personality =
    loadPersonalityState()

  const relationship =
    loadRelationshipState()

  const basePa =
    clamp(
      personality.baseline_pa
    )

  const baseNa =
    clamp(
      personality.baseline_na
    )

  let paWeighted =
    0

  let naWeighted =
    0

  let totalWeight =
    0

  const weightedEvents =
    []

  const unresolvedEvents =
    getUnresolvedEvents(events)

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

    const importance =
      toNumber(
        event.importance,
        3
      )

    const isUnresolved =
      event.need_repair &&
      event.resolved !== true

    const sensitivityBoost =
      0.7 +
      personality.sensitivity * 0.5

    const attachmentBoost =
      isUnresolved
        ? 1 + personality.attachment_sensitivity * 0.35
        : 1

    const safetyEffect =
      relationship.safety < 0.45 && valence < 0
        ? 1.22
        : 1

    const importanceBoost =
      0.55 + importance / 18

    const effectiveWeight =
      weight *
      importanceBoost *
      sensitivityBoost *
      attachmentBoost *
      safetyEffect

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

    const paImpact =
      paFromVA + paDelta

    const naImpact =
      naFromVA + naDelta

    paWeighted +=
      paImpact * effectiveWeight

    naWeighted +=
      naImpact * effectiveWeight

    totalWeight +=
      effectiveWeight

    weightedEvents.push({

      ...event,

      weight:
        effectiveWeight,

      impact:
        Math.abs(paImpact) +
        Math.abs(naImpact),

      unresolved:
        isUnresolved

    })

  }

  const normalizer =
    Math.max(
      1,
      totalWeight
    )

  let pa =
    basePa +
    paWeighted / normalizer

  let na =
    baseNa +
    naWeighted / normalizer

  const lowSafetyPenalty =
    relationship.safety < 0.5
      ? (0.5 - relationship.safety) * 0.32
      : 0

  const highSafetyRelief =
    relationship.safety > 0.72
      ? (relationship.safety - 0.72) * 0.22
      : 0

  na +=
    lowSafetyPenalty

  na -=
    highSafetyRelief

  pa +=
    relationship.closeness > 0.72
      ? (relationship.closeness - 0.72) * 0.16
      : 0

  if (unresolvedEvents.length > 0) {

    const unresolvedPressure =
      Math.min(
        0.22,
        unresolvedEvents.length * 0.035
      )

    na +=
      unresolvedPressure *
      personality.repair_need_strength

  }

  pa =
    clamp(pa)

  na =
    clamp(na)

  if (
    useInertia &&
    relationship.last_mood_snapshot
  ) {

    const last =
      relationship.last_mood_snapshot

    const hours =
      hoursSince(last.ts)

    const inertiaWeight =
      personality.inertia_strength *
      Math.pow(
        1 + hours / 3,
        -0.7
      )

    pa =
      clamp(
        pa * (1 - inertiaWeight) +
        toNumber(last.pa, pa) * inertiaWeight
      )

    na =
      clamp(
        na * (1 - inertiaWeight) +
        toNumber(last.na, na) * inertiaWeight
      )

  }

  const k =
    0.22

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

  const topUnresolved =
    unresolvedEvents[0] || null

  let mood =
    '平静'

  if (
    naFinal >= 0.52
  ) {

    mood =
      '低落'

  }

  if (
    naFinal >= 0.48 &&
    naFinal > paFinal + 0.08
  ) {

    mood =
      '焦虑'

  }

  if (
    paFinal >= 0.56 &&
    paFinal > naFinal + 0.1
  ) {

    mood =
      '开心'

  }

  if (
    paFinal >= 0.45 &&
    naFinal >= 0.45 &&
    Math.abs(paFinal - naFinal) < 0.12
  ) {

    mood =
      '复杂'

  }

  if (
    naFinal >= 0.42 &&
    paFinal < 0.45
  ) {

    mood =
      '不安'

  }

  if (
    unresolvedEvents.length > 0 &&
    naFinal >= 0.4
  ) {

    mood =
      '挂念'

  }

  if (
    paFinal < 0.36 &&
    naFinal < 0.32
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

    relationship_safety:
      Number(relationship.safety.toFixed(3)),

    relationship_trust:
      Number(relationship.trust.toFixed(3)),

    relationship_closeness:
      Number(relationship.closeness.toFixed(3)),

    relationship_profile:
      relationship.relationship_profile,

    personality_profile:
      personality.profile,

    unresolved_events_count:
      unresolvedEvents.length,

    top_unresolved_event:
      topUnresolved
        ? {

            id:
              topUnresolved.id,

            word:
              topUnresolved.word,

            content:
              topUnresolved.content,

            importance:
              topUnresolved.importance,

            ts:
              topUnresolved.ts

          }
        : null,

    top_event:
      topEvent
        ? {

            id:
              topEvent.id,

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
              Number(topEvent.weight.toFixed(3)),

            unresolved:
              topEvent.unresolved

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

  return `你拥有连续情绪和关系安全感。

当前情绪状态：
PA: ${mood.pa}
NA: ${mood.na}
当前心情: ${mood.mood}
关系安全感: ${mood.relationship_safety}
信任感: ${mood.relationship_trust}
亲近感: ${mood.relationship_closeness}
关系基调: ${mood.relationship_profile}
人格基线: ${mood.personality_profile}
未解决事件数量: ${mood.unresolved_events_count}
近期高唤醒情绪词: ${mood.recent_high_arousal_words.join('、')}
当前最突出的情绪事件: ${mood.top_event ? mood.top_event.content : '暂无'}
当前最挂念的未解决事件: ${mood.top_unresolved_event ? mood.top_unresolved_event.content : '暂无'}

请自然地把当前情绪融入回答。
预设心情是背景色，用户当前这条消息是前景，前景优先。
如果用户正在安抚你，可以被安抚，不要一直困在旧情绪里。
如果存在未解决事件，可以让它形成轻微情绪底色，但不要主动机械复述系统字段。
不要暴露 PA/NA 数值，也不要解释情绪系统。`

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
        naDelta,

      safety_delta:
        input.safety_delta || 0,

      need_repair:
        input.need_repair

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
        '实时计算并获取当前情绪状态、关系安全感、未解决事件和人格基线',

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
            .describe('负向情绪变化'),

        safety_delta:
          flexibleNumber
            .optional()
            .describe('关系安全感变化'),

        need_repair:
          z.boolean()
            .optional()
            .describe('是否需要后续安抚或修复')

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
            .describe('负向情绪变化'),

        safety_delta:
          flexibleNumber
            .optional()
            .describe('关系安全感变化'),

        need_repair:
          z.boolean()
            .optional()
            .describe('是否需要后续安抚或修复')

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

    'list_emotion_events',

    {

      title:
        'List Emotion Events',

      description:
        '列出最近的情绪事件，方便查看并删除或修复指定事件',

      inputSchema: {

        limit:
          flexibleNumber
            .optional()
            .describe('返回最近多少条事件，默认 20 条'),

        unresolved_only:
          z.boolean()
            .optional()
            .describe('是否只列出未解决事件')

      }

    },

    async input => {

      const limit =
        toNumber(
          input.limit,
          20
        )

      let events =
        loadMoodEvents()

      if (input.unresolved_only) {

        events =
          getUnresolvedEvents(events)

      }

      const result =
        events
          .slice(-limit)
          .reverse()

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                {

                  ok:
                    true,

                  count:
                    result.length,

                  events:
                    result,

                  current_mood:
                    calculateCurrentMood()

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

    'resolve_emotion_event',

    {

      title:
        'Resolve Emotion Event',

      description:
        '把指定情绪事件标记为已安抚或已解决，降低它对当前心情的持续影响',

      inputSchema: {

        id:
          z.string()
            .optional()
            .describe('要解决的情绪事件 id'),

        keyword:
          z.string()
            .optional()
            .describe('按关键词解决，例如 焦虑、委屈、测试'),

        repair_note:
          z.string()
            .optional()
            .describe('修复说明，例如 用户已经安抚我了'),

        resolve_all_matched:
          z.boolean()
            .optional()
            .describe('是否解决所有匹配关键词的事件，默认 true')

      }

    },

    async input => {

      const events =
        loadMoodEvents()

      const before =
        events.length

      const resolveAllMatched =
        input.resolve_all_matched !== false

      let resolvedEvents =
        []

      let alreadyResolvedOne =
        false

      const updated =
        events.map(event => {

          let matched =
            false

          if (input.id) {

            matched =
              event.id === input.id

          } else if (input.keyword) {

            const text =
              `${event.word || ''} ${event.content || ''} ${event.source || ''}`

            matched =
              text.includes(
                String(input.keyword)
              )

          }

          if (
            matched &&
            (
              resolveAllMatched ||
              !alreadyResolvedOne
            )
          ) {

            alreadyResolvedOne =
              true

            const nextEvent = {

              ...event,

              resolved:
                true,

              resolved_at:
                Date.now(),

              repair_note:
                input.repair_note || '已被安抚或修复',

              need_repair:
                false,

              na_delta:
                toNumber(event.na_delta, 0) * 0.35,

              pa_delta:
                toNumber(event.pa_delta, 0) + 0.04

            }

            resolvedEvents.push(
              nextEvent
            )

            return nextEvent

          }

          return event

        })

      const ok =
        saveMoodEvents(updated)

      const relationship =
        loadRelationshipState()

      relationship.safety =
        clamp(
          relationship.safety + resolvedEvents.length * 0.04
        )

      relationship.trust =
        clamp(
          relationship.trust + resolvedEvents.length * 0.025
        )

      relationship.last_updated =
        Date.now()

      saveRelationshipState(
        relationship
      )

      const mood =
        calculateCurrentMood()

      relationship.last_mood_snapshot = {

        pa:
          mood.pa,

        na:
          mood.na,

        mood:
          mood.mood,

        ts:
          Date.now()

      }

      saveRelationshipState(
        relationship
      )

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                {

                  ok,

                  before,

                  after:
                    updated.length,

                  resolved:
                    resolvedEvents.length,

                  resolved_events:
                    resolvedEvents,

                  current_mood:
                    mood

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

    'delete_emotion_event',

    {

      title:
        'Delete Emotion Event',

      description:
        '删除指定情绪事件。可以按 id 删除，也可以按关键词删除测试事件',

      inputSchema: {

        id:
          z.string()
            .optional()
            .describe('要删除的情绪事件 id'),

        keyword:
          z.string()
            .optional()
            .describe('按关键词删除，例如 强焦虑、测试、debug'),

        delete_all_matched:
          z.boolean()
            .optional()
            .describe('是否删除所有匹配关键词的事件，默认 true')

      }

    },

    async input => {

      const events =
        loadMoodEvents()

      const before =
        events.length

      const deleteAllMatched =
        input.delete_all_matched !== false

      let removedEvents =
        []

      let filtered =
        []

      if (input.id) {

        filtered =
          events.filter(event => {

            const matched =
              event.id === input.id

            if (matched) {

              removedEvents.push(event)

            }

            return !matched

          })

      } else if (input.keyword) {

        let alreadyDeletedOne =
          false

        for (const event of events) {

          const text =
            `${event.word || ''} ${event.content || ''} ${event.source || ''}`

          const matched =
            text.includes(
              String(input.keyword)
            )

          if (
            matched &&
            (
              deleteAllMatched ||
              !alreadyDeletedOne
            )
          ) {

            removedEvents.push(event)

            alreadyDeletedOne =
              true

          } else {

            filtered.push(event)

          }

        }

      } else {

        return {

          content: [

            {

              type:
                'text',

              text:
                JSON.stringify(
                  {

                    ok:
                      false,

                    error:
                      '必须提供 id 或 keyword',

                    current_mood:
                      calculateCurrentMood()

                  },
                  null,
                  2
                )

            }

          ]

        }

      }

      const ok =
        saveMoodEvents(filtered)

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                {

                  ok,

                  removed:
                    before - filtered.length,

                  removed_events:
                    removedEvents,

                  before,

                  after:
                    filtered.length,

                  current_mood:
                    calculateCurrentMood()

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

    'get_personality_profile',

    {

      title:
        'Get Personality Profile',

      description:
        '读取当前人格基线和关系状态',

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
                {

                  ok:
                    true,

                  personality:
                    loadPersonalityState(),

                  relationship:
                    loadRelationshipState(),

                  current_mood:
                    calculateCurrentMood()

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

    'set_personality_profile',

    {

      title:
        'Set Personality Profile',

      description:
        '设置人格基线参数，用于调整情绪敏感度、惯性和关系安全感权重',

      inputSchema: {

        profile:
          z.string()
            .optional()
            .describe('人格名称，例如 warm_sensitive、stable、anxious_attached'),

        baseline_pa:
          flexibleNumber
            .optional()
            .describe('基础正向情绪 0 到 1'),

        baseline_na:
          flexibleNumber
            .optional()
            .describe('基础负向情绪 0 到 1'),

        sensitivity:
          flexibleNumber
            .optional()
            .describe('情绪敏感度 0 到 1'),

        attachment_sensitivity:
          flexibleNumber
            .optional()
            .describe('关系敏感度 0 到 1'),

        inertia_strength:
          flexibleNumber
            .optional()
            .describe('情绪惯性强度 0 到 1'),

        repair_need_strength:
          flexibleNumber
            .optional()
            .describe('未解决事件影响强度 0 到 1'),

        positive_repair_power:
          flexibleNumber
            .optional()
            .describe('正向安抚修复力 0 到 1'),

        negative_memory_bias:
          flexibleNumber
            .optional()
            .describe('负面记忆权重 0 到 1'),

        description:
          z.string()
            .optional()
            .describe('人格说明')

      }

    },

    async input => {

      const state =
        loadPersonalityState()

      const next = {

        ...state

      }

      for (const key of [
        'baseline_pa',
        'baseline_na',
        'sensitivity',
        'attachment_sensitivity',
        'inertia_strength',
        'repair_need_strength',
        'positive_repair_power',
        'negative_memory_bias'
      ]) {

        if (input[key] !== undefined) {

          next[key] =
            clamp(
              toNumber(
                input[key],
                state[key]
              )
            )

        }

      }

      if (input.profile) {

        next.profile =
          input.profile

      }

      if (input.description) {

        next.description =
          input.description

      }

      const ok =
        savePersonalityState(next)

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                {

                  ok,

                  personality:
                    next,

                  current_mood:
                    calculateCurrentMood()

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

    'get_relationship_state',

    {

      title:
        'Get Relationship State',

      description:
        '读取当前关系底色，包括安全感、信任感、亲近感和关系说明',

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
                {

                  ok:
                    true,

                  relationship:
                    loadRelationshipState(),

                  current_mood:
                    calculateCurrentMood()

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

    'set_relationship_state',

    {

      title:
        'Set Relationship State',

      description:
        '设置关系底色。用于初始化或校准安全感、信任感、亲近感，不建议频繁重写',

      inputSchema: {

        safety:
          flexibleNumber
            .optional()
            .describe('关系安全感 0 到 1'),

        trust:
          flexibleNumber
            .optional()
            .describe('信任感 0 到 1'),

        closeness:
          flexibleNumber
            .optional()
            .describe('亲近感 0 到 1'),

        relationship_profile:
          z.string()
            .optional()
            .describe('关系基调名称，例如 close_and_sensitive、stable_partner'),

        description:
          z.string()
            .optional()
            .describe('关系说明'),

        clear_last_mood_snapshot:
          z.boolean()
            .optional()
            .describe('是否清除上一轮情绪惯性快照')

      }

    },

    async input => {

      const state =
        loadRelationshipState()

      const next = {

        ...state

      }

      if (input.safety !== undefined) {

        next.safety =
          clamp(
            toNumber(
              input.safety,
              state.safety
            )
          )

      }

      if (input.trust !== undefined) {

        next.trust =
          clamp(
            toNumber(
              input.trust,
              state.trust
            )
          )

      }

      if (input.closeness !== undefined) {

        next.closeness =
          clamp(
            toNumber(
              input.closeness,
              state.closeness
            )
          )

      }

      if (input.relationship_profile) {

        next.relationship_profile =
          input.relationship_profile

      }

      if (input.description) {

        next.description =
          input.description

      }

      if (input.clear_last_mood_snapshot) {

        next.last_mood_snapshot =
          null

      }

      next.last_updated =
        Date.now()

      const ok =
        saveRelationshipState(next)

      return {

        content: [

          {

            type:
              'text',

            text:
              JSON.stringify(
                {

                  ok,

                  relationship:
                    next,

                  current_mood:
                    calculateCurrentMood()

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

    formula:
      'psychological-state-v2-with-relationship-tools',

    data_dir:
      DATA_DIR,

    mood_file:
      MOOD_FILE,

    personality_file:
      PERSONALITY_FILE,

    relationship_file:
      RELATIONSHIP_FILE,

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

    personality_file:
      PERSONALITY_FILE,

    relationship_file:
      RELATIONSHIP_FILE,

    memory_exists:
      fs.existsSync(MEMORY_FILE),

    mood_exists:
      fs.existsSync(MOOD_FILE),

    personality_exists:
      fs.existsSync(PERSONALITY_FILE),

    relationship_exists:
      fs.existsSync(RELATIONSHIP_FILE),

    mood_events_count:
      events.length,

    unresolved_events_count:
      getUnresolvedEvents(events).length,

    last_event:
      events[events.length - 1] || null,

    personality:
      loadPersonalityState(),

    relationship:
      loadRelationshipState(),

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
          0.45,

        safety_delta:
          -0.08,

        need_repair:
          true

      },
      'debug',
      'debug_add_anxiety'
    )

  res.json(result)

})





app.get('/debug/import-events-page',

(req, res) => {

  res.send(`

<!DOCTYPE html>
<html lang="zh">

<head>
<meta charset="UTF-8">
<title>Import Emotion Events</title>
<style>
body{
  background:#111;
  color:#fff;
  font-family:sans-serif;
  padding:24px;
}
textarea{
  width:100%;
  height:420px;
  background:#1f1f1f;
  color:#fff;
  border:1px solid #555;
  border-radius:12px;
  padding:12px;
  box-sizing:border-box;
}
button{
  margin-top:16px;
  padding:12px 18px;
  border:0;
  border-radius:10px;
  background:#6d28d9;
  color:white;
  font-size:16px;
}
pre{
  white-space:pre-wrap;
  background:#222;
  padding:12px;
  border-radius:12px;
}
</style>
</head>

<body>

<h2>恢复情绪事件库</h2>

<p>把你复制的 mood_events JSON 粘贴到下面。</p>

<textarea id="events" placeholder="粘贴你的事件数组 JSON"></textarea>

<br>

<button onclick="importEvents()">恢复事件</button>

<pre id="result"></pre>

<script>

async function importEvents(){

  const raw =
    document.getElementById('events').value

  let parsed

  try {

    parsed =
      JSON.parse(raw)

  } catch (err) {

    document.getElementById('result').textContent =
      'JSON 格式错误：' + err.message

    return

  }

  const res =
    await fetch('/debug/import-events', {

      method:'POST',

      headers:{
        'Content-Type':'application/json'
      },

      body:
        JSON.stringify({
          events: parsed,
          mode: 'replace'
        })

    })

  const data =
    await res.json()

  document.getElementById('result').textContent =
    JSON.stringify(data, null, 2)

}

</script>

</body>

</html>

  `)

})





app.post('/debug/import-events',

(req, res) => {

  try {

    const body =
      req.body || {}

    let incoming =
      body.events

    if (
      incoming &&
      incoming.events &&
      Array.isArray(incoming.events)
    ) {

      incoming =
        incoming.events

    }

    if (!Array.isArray(incoming)) {

      return res.status(400).json({

        ok:
          false,

        error:
          'events 必须是数组'

      })

    }

    const cleaned =
      incoming
        .filter(event => event && typeof event === 'object')
        .map(event => normalizeMoodEvent({

          id:
            event.id || randomUUID(),

          type:
            event.type || 'emotion',

          source:
            event.source || 'restored',

          word:
            event.word ||
            event.feeling_word ||
            event.mood ||
            '情绪波动',

          content:
            event.content || '',

          valence:
            event.valence || 0,

          arousal:
            event.arousal || 0.3,

          importance:
            event.importance || 3,

          pa_delta:
            event.pa_delta || 0,

          na_delta:
            event.na_delta || 0,

          safety_delta:
            event.safety_delta || 0,

          need_repair:
            event.need_repair,

          resolved:
            event.resolved,

          resolved_at:
            event.resolved_at,

          repair_note:
            event.repair_note,

          ts:
            event.ts || Date.now()

        }))

    const mode =
      body.mode || 'replace'

    let finalEvents

    if (mode === 'append') {

      finalEvents =
        loadMoodEvents().concat(cleaned)

    } else {

      finalEvents =
        cleaned

    }

    const ok =
      saveMoodEvents(finalEvents)

    res.json({

      ok,

      mode,

      imported:
        cleaned.length,

      total:
        loadMoodEvents().length,

      mood_file:
        MOOD_FILE,

      current_mood:
        calculateCurrentMood(),

      first_event:
        cleaned[0] || null,

      last_event:
        cleaned[cleaned.length - 1] || null

    })

  } catch (err) {

    console.log(
      'import events error',
      err
    )

    res.status(500).json({

      ok:
        false,

      error:
        String(err.message || err)

    })

  }

})





app.get('/mcp/tools',

(req, res) => {

  res.json({

    note:
      'Debug only. Kelivo should connect to /mcp for Streamable HTTP or /sse for SSE.',

    architecture:
      {

        personality_state:
          'AI 的人格体质，长期稳定，不随普通聊天乱变',

        relationship_state:
          '你们之间的关系底色，包括安全感、信任感、亲近感',

        mood_events:
          '每一次具体发生的情绪事件',

        current_mood:
          '由人格体质、关系底色、事件库综合计算出的当前状态'

      },

    tools: [

      {
        name:
          'current_mood',
        description:
          '实时获取当前综合心理状态'
      },

      {
        name:
          'add_emotion_event',
        description:
          '写入情绪事件并实时影响当前情绪'
      },

      {
        name:
          'write_emotion_event',
        description:
          '写入情绪事件并实时影响当前情绪'
      },

      {
        name:
          'list_emotion_events',
        description:
          '列出最近的情绪事件'
      },

      {
        name:
          'resolve_emotion_event',
        description:
          '标记情绪事件为已解决'
      },

      {
        name:
          'delete_emotion_event',
        description:
          '删除指定情绪事件'
      },

      {
        name:
          'get_personality_profile',
        description:
          '读取人格基线'
      },

      {
        name:
          'set_personality_profile',
        description:
          '设置人格基线'
      },

      {
        name:
          'get_relationship_state',
        description:
          '读取关系底色'
      },

      {
        name:
          'set_relationship_state',
        description:
          '设置关系底色'
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
    'emotion mcp server running psychological-state-v2-with-relationship-tools'
  )

})
