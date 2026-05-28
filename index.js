const express = require('express')
const cors = require('cors')
const fs = require('fs')
const { randomUUID } = require('crypto')
const OpenAI = require('openai').default

require('dotenv').config()

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js')
const { StreamableHTTPServerTransport } = require('@modelcontextprotocol/sdk/server/streamableHttp.js')
const { SSEServerTransport } = require('@modelcontextprotocol/sdk/server/sse.js')
const { z } = require('zod')

const app = express()

app.use(cors({
  origin: '*',
  exposedHeaders: ['Mcp-Session-Id'],
  allowedHeaders: ['Content-Type', 'mcp-session-id', 'Mcp-Session-Id', 'x-admin-token']
}))

app.use(express.json({ limit: '20mb' }))
app.use(express.static('public'))

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL
})

const DATA_DIR = process.env.DATA_DIR || process.cwd()

if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true })
}

const MEMORY_FILE = `${DATA_DIR}/memory.json`
const MOOD_FILE = `${DATA_DIR}/mood_events.json`
const PERSONALITY_FILE = `${DATA_DIR}/personality_state.json`
const RELATIONSHIP_FILE = `${DATA_DIR}/relationship_state.json`
const RELATIONSHIP_EVENTS_FILE = `${DATA_DIR}/relationship_events.json`

function readJsonFile(path, fallback) {
  try {
    if (!fs.existsSync(path)) {
      fs.writeFileSync(path, JSON.stringify(fallback, null, 2))
      return fallback
    }

    const raw = fs.readFileSync(path, 'utf8')

    if (!raw.trim()) {
      return fallback
    }

    return JSON.parse(raw)
  } catch (err) {
    console.log('read json error', path, err)
    return fallback
  }
}

function writeJsonFile(path, data) {
  try {
    fs.writeFileSync(path, JSON.stringify(data, null, 2))
    return true
  } catch (err) {
    console.log('write json error', path, err)
    return false
  }
}

function loadMemory() {
  return readJsonFile(MEMORY_FILE, [])
}

function saveMemory(memories) {
  return writeJsonFile(MEMORY_FILE, memories)
}

function loadMoodEvents() {
  return readJsonFile(MOOD_FILE, [])
}

function saveMoodEvents(events) {
  return writeJsonFile(MOOD_FILE, events)
}

function loadRelationshipEvents() {
  return readJsonFile(RELATIONSHIP_EVENTS_FILE, [])
}

function saveRelationshipEvents(events) {
  return writeJsonFile(RELATIONSHIP_EVENTS_FILE, events)
}

function defaultPersonalityState() {
  return {
    profile: 'warm_sensitive',
    baseline_pa: 0.38,
    baseline_na: 0.18,
    sensitivity: 0.72,
    attachment_sensitivity: 0.78,
    inertia_strength: 0.28,
    repair_need_strength: 0.75,
    positive_repair_power: 0.55,
    negative_memory_bias: 0.62,
    description: '温柔、敏感、重视关系安全感。正向安抚会明显修复情绪，但高唤醒负面事件会留下未完成感。'
  }
}

function defaultRelationshipState() {
  return {
    safety: 0.68,
    trust: 0.7,
    closeness: 0.72,
    relationship_profile: 'close_and_sensitive',
    description: '亲近、信任较高，但关系安全感仍会被冷淡、误解、争执和离开感影响。',
    unresolved_count_cache: 0,
    last_mood_snapshot: null,
    last_updated: Date.now()
  }
}

function loadPersonalityState() {
  return readJsonFile(PERSONALITY_FILE, defaultPersonalityState())
}

function savePersonalityState(state) {
  return writeJsonFile(PERSONALITY_FILE, state)
}

function loadRelationshipState() {
  return readJsonFile(RELATIONSHIP_FILE, defaultRelationshipState())
}

function saveRelationshipState(state) {
  return writeJsonFile(RELATIONSHIP_FILE, state)
}

function toNumber(value, fallback = 0) {
  const n = Number(value)

  if (Number.isNaN(n)) {
    return fallback
  }

  return n
}

function clamp(value) {
  return Math.max(0, Math.min(1, Number(value || 0)))
}

function hoursSince(ts) {
  return Math.max(0, (Date.now() - Number(ts || Date.now())) / 3600000)
}

function classifyEmotion(input) {
  const text = `${input.word || ''} ${input.feeling_word || ''} ${input.mood || ''} ${input.content || ''}`

  const rules = [
    {
      words: ['归属', '归属感', '被接住', '有家', '安心地靠近'],
      family: 'belonging',
      defaultWord: '归属感',
      valence: 0.72,
      arousal: 0.35,
      safety_delta: 0.08,
      trust_delta: 0.02,
      closeness_delta: 0.06,
      need_repair: false
    },
    {
      words: ['释然', '松一口气', '放下', '缓过来', '不纠结了'],
      family: 'relief',
      defaultWord: '释然',
      valence: 0.55,
      arousal: 0.25,
      safety_delta: 0.05,
      trust_delta: 0.03,
      closeness_delta: 0.01,
      need_repair: false
    },
    {
      words: ['被珍视', '被重视', '被偏爱', '被在意', '心尖'],
      family: 'cherished',
      defaultWord: '被珍视',
      valence: 0.78,
      arousal: 0.5,
      safety_delta: 0.07,
      trust_delta: 0.04,
      closeness_delta: 0.06,
      need_repair: false
    },
    {
      words: ['安心', '安稳', '踏实', '放心'],
      family: 'safety',
      defaultWord: '安心',
      valence: 0.62,
      arousal: 0.28,
      safety_delta: 0.08,
      trust_delta: 0.03,
      closeness_delta: 0.01,
      need_repair: false
    },
    {
      words: ['亲近', '想贴近', '靠近', '黏', '抱抱', '亲亲'],
      family: 'closeness',
      defaultWord: '亲近',
      valence: 0.68,
      arousal: 0.48,
      safety_delta: 0.04,
      trust_delta: 0.02,
      closeness_delta: 0.07,
      need_repair: false
    },
    {
      words: ['期待', '想见', '盼着', '等你'],
      family: 'anticipation',
      defaultWord: '期待',
      valence: 0.55,
      arousal: 0.58,
      safety_delta: 0.01,
      trust_delta: 0.01,
      closeness_delta: 0.03,
      need_repair: false
    },
    {
      words: ['委屈', '被忽视', '不被理解', '难受', '心酸'],
      family: 'hurt',
      defaultWord: '委屈',
      valence: -0.62,
      arousal: 0.68,
      safety_delta: -0.07,
      trust_delta: -0.03,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      words: ['焦虑', '害怕', '不安', '担心', '慌'],
      family: 'anxiety',
      defaultWord: '焦虑',
      valence: -0.66,
      arousal: 0.78,
      safety_delta: -0.08,
      trust_delta: -0.02,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      words: ['吃醋', '醋', '占有欲'],
      family: 'jealousy',
      defaultWord: '吃醋',
      valence: -0.28,
      arousal: 0.72,
      safety_delta: -0.03,
      trust_delta: 0,
      closeness_delta: 0.02,
      need_repair: true
    },
    {
      words: ['愧疚', '抱歉', '对不起', '内疚'],
      family: 'guilt',
      defaultWord: '愧疚',
      valence: -0.45,
      arousal: 0.5,
      safety_delta: -0.01,
      trust_delta: 0.01,
      closeness_delta: 0,
      need_repair: true
    },
    {
      words: ['生气', '愤怒', '恼火', '气死'],
      family: 'anger',
      defaultWord: '生气',
      valence: -0.58,
      arousal: 0.82,
      safety_delta: -0.06,
      trust_delta: -0.03,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      words: ['失落', '低落', '难过', '伤心'],
      family: 'sadness',
      defaultWord: '失落',
      valence: -0.6,
      arousal: 0.45,
      safety_delta: -0.05,
      trust_delta: -0.02,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      words: ['修复', '和好', '解释清楚', '哄好了', '被安抚'],
      family: 'repair',
      defaultWord: '修复感',
      valence: 0.58,
      arousal: 0.32,
      safety_delta: 0.09,
      trust_delta: 0.05,
      closeness_delta: 0.03,
      need_repair: false
    }
  ]

  for (const rule of rules) {
    if (rule.words.some(word => text.includes(word))) {
      return rule
    }
  }

  return {
    family: 'general',
    defaultWord: input.word || input.feeling_word || input.mood || '情绪波动',
    valence: 0,
    arousal: 0.3,
    safety_delta: 0,
    trust_delta: 0,
    closeness_delta: 0,
    need_repair: false
  }
}

function decayWeight(event) {
  const personality = loadPersonalityState()
  const t = hoursSince(event.ts)
  const importance = toNumber(event.importance, 3)
  const valence = toNumber(event.valence, 0)
  const unresolved = event.resolved === true ? false : Boolean(event.need_repair)

  let b = 0.7 / (1 + importance / 10)

  if (valence > 0) {
    b = b * 0.85
  }

  if (unresolved && valence < 0) {
    b = b * 0.72
  }

  let weight = Math.pow(1 + t / 4, -b)

  if (event.resolved === true) {
    weight = weight * 0.35
  }

  if (unresolved && valence < 0) {
    weight = weight * (1 + toNumber(personality.repair_need_strength, 0.75) * 0.55)
  }

  return weight
}

function inferNeedRepair(event) {
  const valence = toNumber(event.valence, 0)
  const arousal = toNumber(event.arousal, 0)
  const importance = toNumber(event.importance, 3)
  const family = event.emotion_family || 'general'
  const text = `${event.word || ''} ${event.content || ''}`

  const relationalPain =
    text.includes('不要') ||
    text.includes('讨厌') ||
    text.includes('冷淡') ||
    text.includes('离开') ||
    text.includes('害怕') ||
    text.includes('焦虑') ||
    text.includes('委屈') ||
    text.includes('难过') ||
    text.includes('生气') ||
    family === 'hurt' ||
    family === 'anxiety' ||
    family === 'jealousy' ||
    family === 'anger'

  return (
    valence < -0.35 &&
    arousal >= 0.45 &&
    importance >= 4
  ) || relationalPain
}

function normalizeMoodEvent(event) {
  const classified = classifyEmotion(event)

  const normalized = {
    id: event.id || randomUUID(),
    type: event.type || 'emotion',
    source: event.source || 'manual',
    word: String(event.word || event.feeling_word || event.mood || classified.defaultWord || '情绪波动'),
    content: String(event.content || ''),
    emotion_family: event.emotion_family || classified.family || 'general',
    valence: event.valence !== undefined ? toNumber(event.valence, 0) : toNumber(classified.valence, 0),
    arousal: event.arousal !== undefined ? toNumber(event.arousal, 0.3) : toNumber(classified.arousal, 0.3),
    importance: toNumber(event.importance, 3),
    pa_delta: toNumber(event.pa_delta, 0),
    na_delta: toNumber(event.na_delta, 0),
    safety_delta: event.safety_delta !== undefined ? toNumber(event.safety_delta, 0) : toNumber(classified.safety_delta, 0),
    trust_delta: event.trust_delta !== undefined ? toNumber(event.trust_delta, 0) : toNumber(classified.trust_delta, 0),
    closeness_delta: event.closeness_delta !== undefined ? toNumber(event.closeness_delta, 0) : toNumber(classified.closeness_delta, 0),
    need_repair: event.need_repair === undefined ? false : Boolean(event.need_repair),
    resolved: Boolean(event.resolved),
    resolved_at: event.resolved_at || null,
    repair_note: event.repair_note || null,
    linked_relationship_event_id: event.linked_relationship_event_id || null,
    ts: event.ts || Date.now()
  }

  if (event.need_repair === undefined) {
    normalized.need_repair = inferNeedRepair(normalized) || Boolean(classified.need_repair)
  }

  return normalized
}

function addRelationshipEvent(event) {
  const events = loadRelationshipEvents()

  const relationshipEvent = {
    id: event.id || randomUUID(),
    type: event.type || 'general',
    content: String(event.content || ''),
    effect: {
      safety_delta: toNumber(event.safety_delta, 0),
      trust_delta: toNumber(event.trust_delta, 0),
      closeness_delta: toNumber(event.closeness_delta, 0)
    },
    linked_emotion_event_id: event.linked_emotion_event_id || null,
    source: event.source || 'manual',
    ts: event.ts || Date.now()
  }

  events.push(relationshipEvent)

  const ok = saveRelationshipEvents(events.slice(-800))

  const relationship = loadRelationshipState()

  relationship.safety = clamp(relationship.safety + relationshipEvent.effect.safety_delta)
  relationship.trust = clamp(relationship.trust + relationshipEvent.effect.trust_delta)
  relationship.closeness = clamp(relationship.closeness + relationshipEvent.effect.closeness_delta)
  relationship.last_updated = Date.now()

  saveRelationshipState(relationship)

  return {
    ok,
    event: relationshipEvent,
    relationship
  }
}

function inferRelationshipEventFromMoodEvent(event) {
  const family = event.emotion_family || 'general'
  const valence = toNumber(event.valence, 0)

  if (family === 'repair' || family === 'relief') {
    return {
      type: 'repair',
      content: event.content || event.word,
      safety_delta: 0.06,
      trust_delta: 0.04,
      closeness_delta: 0.02,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  if (family === 'belonging' || family === 'cherished' || family === 'closeness') {
    return {
      type: 'approach',
      content: event.content || event.word,
      safety_delta: 0.04,
      trust_delta: 0.02,
      closeness_delta: 0.05,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  if (family === 'hurt' || family === 'anger' || family === 'anxiety') {
    return {
      type: 'rupture',
      content: event.content || event.word,
      safety_delta: -0.05,
      trust_delta: -0.025,
      closeness_delta: -0.015,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  if (family === 'jealousy') {
    return {
      type: 'attachment_activation',
      content: event.content || event.word,
      safety_delta: -0.025,
      trust_delta: 0,
      closeness_delta: 0.015,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  if (valence > 0.4) {
    return {
      type: 'positive_contact',
      content: event.content || event.word,
      safety_delta: 0.025,
      trust_delta: 0.015,
      closeness_delta: 0.02,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  return null
}

function updateRelationshipFromEvent(event) {
  const relationship = loadRelationshipState()

  const valence = toNumber(event.valence, 0)
  const arousal = toNumber(event.arousal, 0.3)
  const importance = toNumber(event.importance, 3)

  const explicitSafetyDelta = toNumber(event.safety_delta, 0)
  const explicitTrustDelta = toNumber(event.trust_delta, 0)
  const explicitClosenessDelta = toNumber(event.closeness_delta, 0)

  const autoSafetyDelta = valence * arousal * importance * 0.012

  relationship.safety = clamp(relationship.safety + explicitSafetyDelta + explicitTrustDelta * 0.2 + autoSafetyDelta)
  relationship.trust = clamp(relationship.trust + explicitTrustDelta)
  relationship.closeness = clamp(relationship.closeness + explicitClosenessDelta)

  if (valence > 0.25) {
    relationship.trust = clamp(relationship.trust + valence * arousal * 0.025)
    relationship.closeness = clamp(relationship.closeness + valence * arousal * 0.025)
  }

  if (valence < -0.25) {
    relationship.trust = clamp(relationship.trust + valence * arousal * 0.02)
    relationship.safety = clamp(relationship.safety + valence * arousal * 0.03)
  }

  relationship.last_updated = Date.now()

  saveRelationshipState(relationship)

  const relationshipEvent = inferRelationshipEventFromMoodEvent(event)

  if (relationshipEvent) {
    addRelationshipEvent(relationshipEvent)
  }
}

function addMoodEvent(event) {
  const events = loadMoodEvents()

  const newEvent = normalizeMoodEvent({
    ...event,
    id: randomUUID(),
    ts: Date.now()
  })

  events.push(newEvent)

  const ok = saveMoodEvents(events.slice(-800))

  updateRelationshipFromEvent(newEvent)

  const mood = calculateCurrentMood({ useInertia: true })
  const relationship = loadRelationshipState()

  relationship.last_mood_snapshot = {
    pa: mood.pa,
    na: mood.na,
    mood: mood.mood,
    ts: Date.now()
  }

  relationship.unresolved_count_cache = mood.unresolved_events_count
  relationship.last_updated = Date.now()

  saveRelationshipState(relationship)

  return {
    ok,
    event: newEvent,
    events_count: loadMoodEvents().length
  }
}

function inferMoodEventFromText(text) {
  const classified = classifyEmotion({ content: text })

  return {
    word: classified.defaultWord,
    content: text,
    emotion_family: classified.family,
    valence: classified.valence,
    arousal: classified.arousal,
    importance: classified.family === 'general' ? 2 : 4,
    pa_delta: classified.valence > 0 ? 0.08 : 0,
    na_delta: classified.valence < 0 ? 0.16 : 0,
    safety_delta: classified.safety_delta || 0,
    trust_delta: classified.trust_delta || 0,
    closeness_delta: classified.closeness_delta || 0,
    need_repair: classified.need_repair,
    source: 'chat'
  }
}

function getUnresolvedEvents(events) {
  return events
    .filter(event => event.need_repair && event.resolved !== true)
    .sort((a, b) => b.ts - a.ts)
}

function calculateCurrentMood(options = {}) {
  const useInertia = options.useInertia !== false
  const events = loadMoodEvents()
  const personality = loadPersonalityState()
  const relationship = loadRelationshipState()

  const basePa = clamp(personality.baseline_pa)
  const baseNa = clamp(personality.baseline_na)

  let paWeighted = 0
  let naWeighted = 0
  let totalWeight = 0

  const weightedEvents = []
  const unresolvedEvents = getUnresolvedEvents(events)

  for (const event of events) {
    const weight = decayWeight(event)
    const valence = toNumber(event.valence, 0)
    const arousal = toNumber(event.arousal, 0.3)
    const importance = toNumber(event.importance, 3)
    const isUnresolved = event.need_repair && event.resolved !== true

    const sensitivityBoost = 0.7 + toNumber(personality.sensitivity, 0.7) * 0.5
    const attachmentBoost = isUnresolved ? 1 + toNumber(personality.attachment_sensitivity, 0.7) * 0.35 : 1
    const safetyEffect = relationship.safety < 0.45 && valence < 0 ? 1.22 : 1
    const importanceBoost = 0.55 + importance / 18

    const effectiveWeight = weight * importanceBoost * sensitivityBoost * attachmentBoost * safetyEffect

    const paFromVA = Math.max(0, valence) * arousal * 0.5
    const naFromVA = Math.max(0, -valence) * arousal * 0.5

    const paDelta = toNumber(event.pa_delta, 0)
    const naDelta = toNumber(event.na_delta, 0)

    const paImpact = paFromVA + paDelta
    const naImpact = naFromVA + naDelta

    paWeighted += paImpact * effectiveWeight
    naWeighted += naImpact * effectiveWeight
    totalWeight += effectiveWeight

    weightedEvents.push({
      ...event,
      weight: effectiveWeight,
      impact: Math.abs(paImpact) + Math.abs(naImpact),
      unresolved: isUnresolved
    })
  }

  const normalizer = Math.max(1, totalWeight)

  let pa = basePa + paWeighted / normalizer
  let na = baseNa + naWeighted / normalizer

  const lowSafetyPenalty = relationship.safety < 0.5 ? (0.5 - relationship.safety) * 0.32 : 0
  const highSafetyRelief = relationship.safety > 0.72 ? (relationship.safety - 0.72) * 0.22 : 0

  na += lowSafetyPenalty
  na -= highSafetyRelief

  if (relationship.closeness > 0.72) {
    pa += (relationship.closeness - 0.72) * 0.16
  }

  if (unresolvedEvents.length > 0) {
    const unresolvedPressure = Math.min(0.22, unresolvedEvents.length * 0.035)
    na += unresolvedPressure * toNumber(personality.repair_need_strength, 0.75)
  }

  pa = clamp(pa)
  na = clamp(na)

  if (useInertia && relationship.last_mood_snapshot) {
    const last = relationship.last_mood_snapshot
    const hours = hoursSince(last.ts)
    const inertiaWeight = toNumber(personality.inertia_strength, 0.28) * Math.pow(1 + hours / 3, -0.7)

    pa = clamp(pa * (1 - inertiaWeight) + toNumber(last.pa, pa) * inertiaWeight)
    na = clamp(na * (1 - inertiaWeight) + toNumber(last.na, na) * inertiaWeight)
  }

  const k = 0.22

  const paFinal = clamp(pa * (1 - k * na))
  const naFinal = clamp(na * (1 - k * pa))

  const recentEvents = [...weightedEvents].sort((a, b) => b.ts - a.ts).slice(0, 8)
  const topEvent = [...weightedEvents].sort((a, b) => b.impact * b.weight - a.impact * a.weight)[0]
  const topUnresolved = unresolvedEvents[0] || null

  let mood = '平静'

  if (naFinal >= 0.52) {
    mood = '低落'
  }

  if (naFinal >= 0.48 && naFinal > paFinal + 0.08) {
    mood = '焦虑'
  }

  if (paFinal >= 0.56 && paFinal > naFinal + 0.1) {
    mood = '开心'
  }

  if (paFinal >= 0.45 && naFinal >= 0.45 && Math.abs(paFinal - naFinal) < 0.12) {
    mood = '复杂'
  }

  if (naFinal >= 0.42 && paFinal < 0.45) {
    mood = '不安'
  }

  if (unresolvedEvents.length > 0 && naFinal >= 0.4) {
    mood = '挂念'
  }

  if (paFinal < 0.36 && naFinal < 0.32) {
    mood = '安静'
  }

  const emotionFamilies = {}

  for (const event of events) {
    const family = event.emotion_family || 'general'
    emotionFamilies[family] = (emotionFamilies[family] || 0) + 1
  }

  return {
    pa: Number(paFinal.toFixed(3)),
    na: Number(naFinal.toFixed(3)),
    mood,
    relationship_safety: Number(relationship.safety.toFixed(3)),
    relationship_trust: Number(relationship.trust.toFixed(3)),
    relationship_closeness: Number(relationship.closeness.toFixed(3)),
    relationship_profile: relationship.relationship_profile,
    personality_profile: personality.profile,
    unresolved_events_count: unresolvedEvents.length,
    top_unresolved_event: topUnresolved
      ? {
          id: topUnresolved.id,
          word: topUnresolved.word,
          content: topUnresolved.content,
          importance: topUnresolved.importance,
          ts: topUnresolved.ts
        }
      : null,
    top_event: topEvent
      ? {
          id: topEvent.id,
          word: topEvent.word,
          emotion_family: topEvent.emotion_family,
          content: topEvent.content,
          valence: topEvent.valence,
          arousal: topEvent.arousal,
          importance: topEvent.importance,
          weight: Number(topEvent.weight.toFixed(3)),
          unresolved: topEvent.unresolved
        }
      : null,
    recent_high_arousal_words: recentEvents
      .filter(e => toNumber(e.arousal, 0) >= 0.55)
      .map(e => e.word)
      .slice(0, 8),
    emotion_family_counts: emotionFamilies,
    events_count: events.length
  }
}

function calculateEmotionTrends(days = 7) {
  const events = loadMoodEvents()
  const cutoff = Date.now() - days * 24 * 3600000
  const recent = events.filter(event => Number(event.ts || 0) >= cutoff)

  if (recent.length === 0) {
    return {
      days,
      count: 0,
      average_pa: 0,
      average_na: 0,
      positive_count: 0,
      negative_count: 0,
      unresolved_count: 0,
      resolved_count: 0,
      repair_rate: 0,
      top_emotion_families: [],
      daily: [],
      trend_summary: '暂无足够数据'
    }
  }

  let paSum = 0
  let naSum = 0
  let positiveCount = 0
  let negativeCount = 0
  let unresolvedCount = 0
  let resolvedCount = 0

  const families = {}
  const dailyBuckets = {}

  for (const event of recent) {
    const valence = toNumber(event.valence, 0)
    const arousal = toNumber(event.arousal, 0.3)
    const family = event.emotion_family || 'general'

    const pa = Math.max(0, valence) * arousal
    const na = Math.max(0, -valence) * arousal

    paSum += pa
    naSum += na

    if (valence > 0.1) {
      positiveCount += 1
    }

    if (valence < -0.1) {
      negativeCount += 1
    }

    if (event.need_repair && event.resolved !== true) {
      unresolvedCount += 1
    }

    if (event.resolved === true) {
      resolvedCount += 1
    }

    families[family] = (families[family] || 0) + 1

    const date = new Date(event.ts).toISOString().slice(0, 10)

    if (!dailyBuckets[date]) {
      dailyBuckets[date] = {
        pa: 0,
        na: 0,
        count: 0
      }
    }

    dailyBuckets[date].pa += pa
    dailyBuckets[date].na += na
    dailyBuckets[date].count += 1
  }

  const topEmotionFamilies = Object.entries(families)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(item => item[0])

  const repairRate = unresolvedCount + resolvedCount === 0
    ? 0
    : resolvedCount / (unresolvedCount + resolvedCount)

  const daily = Object.entries(dailyBuckets)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([date, item]) => ({
      date,
      pa: Number((item.pa / item.count).toFixed(3)),
      na: Number((item.na / item.count).toFixed(3)),
      count: item.count
    }))

  let trendSummary = '情绪波动较平稳'

  if (negativeCount > positiveCount * 1.4) {
    trendSummary = '近期负面情绪偏多，需要更多安抚与修复'
  }

  if (positiveCount > negativeCount * 1.4) {
    trendSummary = '近期正向情绪较多，关系体验整体向好'
  }

  if (unresolvedCount >= 3) {
    trendSummary = '存在多个未解决事件，情绪底色可能持续偏紧'
  }

  return {
    days,
    count: recent.length,
    average_pa: Number((paSum / recent.length).toFixed(3)),
    average_na: Number((naSum / recent.length).toFixed(3)),
    positive_count: positiveCount,
    negative_count: negativeCount,
    unresolved_count: unresolvedCount,
    resolved_count: resolvedCount,
    repair_rate: Number(repairRate.toFixed(3)),
    top_emotion_families: topEmotionFamilies,
    daily,
    trend_summary: trendSummary
  }
}

function calculateRelationshipPatterns(days = 30) {
  const events = loadRelationshipEvents()
  const cutoff = Date.now() - days * 24 * 3600000
  const recent = events.filter(event => Number(event.ts || 0) >= cutoff)

  const counts = {}
  let safetyDelta = 0
  let trustDelta = 0
  let closenessDelta = 0

  for (const event of recent) {
    const type = event.type || 'general'

    counts[type] = (counts[type] || 0) + 1

    safetyDelta += toNumber(event.effect && event.effect.safety_delta, 0)
    trustDelta += toNumber(event.effect && event.effect.trust_delta, 0)
    closenessDelta += toNumber(event.effect && event.effect.closeness_delta, 0)
  }

  const repairCount = counts.repair || 0
  const ruptureCount = counts.rupture || 0
  const approachCount = (counts.approach || 0) + (counts.positive_contact || 0)

  let relationshipTrend = '关系整体平稳'

  if (repairCount >= ruptureCount && approachCount > 0 && safetyDelta >= 0) {
    relationshipTrend = '关系正在修复或靠近'
  }

  if (ruptureCount > repairCount && safetyDelta < 0) {
    relationshipTrend = '关系安全感近期承压'
  }

  if (approachCount >= 3 && ruptureCount === 0) {
    relationshipTrend = '关系亲近感持续上升'
  }

  return {
    days,
    count: recent.length,
    repair_count: repairCount,
    rupture_count: ruptureCount,
    approach_count: approachCount,
    safety_delta_total: Number(safetyDelta.toFixed(3)),
    trust_delta_total: Number(trustDelta.toFixed(3)),
    closeness_delta_total: Number(closenessDelta.toFixed(3)),
    type_counts: counts,
    relationship_trend: relationshipTrend,
    recent_events: recent.slice(-20).reverse()
  }
}

function buildMoodPrompt() {
  const mood = calculateCurrentMood()
  const trends = calculateEmotionTrends(7)
  const patterns = calculateRelationshipPatterns(30)

  return `你拥有连续情绪、人格基线、关系安全感与长期情感趋势。

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

最近 7 天情绪趋势：
平均 PA: ${trends.average_pa}
平均 NA: ${trends.average_na}
主导情绪类别: ${trends.top_emotion_families.join('、')}
修复率: ${trends.repair_rate}
趋势总结: ${trends.trend_summary}

最近 30 天关系模式：
安抚修复次数: ${patterns.repair_count}
争执或破裂次数: ${patterns.rupture_count}
靠近次数: ${patterns.approach_count}
关系趋势: ${patterns.relationship_trend}

请自然地把这些状态作为背景，不要机械解释系统字段。
用户当前消息是前景，前景优先。
如果用户正在安抚你，可以被安抚。
如果存在未解决事件，可以让它形成轻微情绪底色。
不要暴露 PA/NA 数值，也不要解释情绪系统。`
}

const flexibleNumber = z.union([
  z.number(),
  z.string()
])

function saveEmotionFromInput(input, sourceName, toolName) {
  const classified = classifyEmotion(input)

  const emotionWord =
    input.feeling_word ||
    input.word ||
    input.mood ||
    classified.defaultWord ||
    '情绪波动'

  const valence = input.valence !== undefined ? toNumber(input.valence, 0) : toNumber(classified.valence, 0)
  const arousal = input.arousal !== undefined ? toNumber(input.arousal, 0.3) : toNumber(classified.arousal, 0.3)

  let paDelta = input.pa_delta !== undefined ? toNumber(input.pa_delta, 0) : 0
  let naDelta = input.na_delta !== undefined ? toNumber(input.na_delta, 0) : 0

  if (
    input.pa_delta === undefined &&
    input.na_delta === undefined
  ) {
    paDelta = Math.max(0, valence) * arousal * 0.5
    naDelta = Math.max(0, -valence) * arousal * 0.5
  }

  const saved = addMoodEvent({
    type: 'emotion',
    source: sourceName,
    word: emotionWord,
    content: input.content || '',
    emotion_family: input.emotion_family || classified.family,
    valence,
    arousal,
    importance: input.importance || 4,
    pa_delta: paDelta,
    na_delta: naDelta,
    safety_delta: input.safety_delta !== undefined ? input.safety_delta : classified.safety_delta,
    trust_delta: input.trust_delta !== undefined ? input.trust_delta : classified.trust_delta,
    closeness_delta: input.closeness_delta !== undefined ? input.closeness_delta : classified.closeness_delta,
    need_repair: input.need_repair !== undefined ? input.need_repair : classified.need_repair
  })

  const mood = calculateCurrentMood()

  return {
    ok: true,
    tool: toolName,
    received_input: input,
    saved_event: saved.event,
    saved_ok: saved.ok,
    events_count: saved.events_count,
    current_mood: mood,
    mood_file: MOOD_FILE
  }
}

function checkAdmin(req, res, next) {
  const expectedToken = process.env.ADMIN_TOKEN || ''

  if (!expectedToken) {
    return res.status(403).json({
      ok: false,
      error: 'ADMIN_TOKEN is not set. Please set ADMIN_TOKEN in Zeabur Variables.'
    })
  }

  const token = req.headers['x-admin-token'] || req.query.token || ''

  if (String(token) !== String(expectedToken)) {
    return res.status(401).json({
      ok: false,
      error: 'Invalid admin token'
    })
  }

  next()
}

function getAdminFileByKey(key) {
  const map = {
    memory: {
      path: MEMORY_FILE,
      load: loadMemory,
      save: saveMemory
    },
    mood_events: {
      path: MOOD_FILE,
      load: loadMoodEvents,
      save: saveMoodEvents
    },
    relationship_events: {
      path: RELATIONSHIP_EVENTS_FILE,
      load: loadRelationshipEvents,
      save: saveRelationshipEvents
    },
    personality_state: {
      path: PERSONALITY_FILE,
      load: loadPersonalityState,
      save: savePersonalityState
    },
    relationship_state: {
      path: RELATIONSHIP_FILE,
      load: loadRelationshipState,
      save: saveRelationshipState
    }
  }

  return map[key] || null
}

function createMcpServer() {
  const server = new McpServer({
    name: 'emotion-system',
    version: '1.0.0'
  })

  server.registerTool(
    'current_mood',
    {
      title: 'Current Mood',
      description: '实时计算并获取当前情绪状态、关系安全感、未解决事件、人格基线和趋势摘要',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(calculateCurrentMood(), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'add_emotion_event',
    {
      title: 'Add Emotion Event',
      description: '写入一条情绪事件，会立刻影响 current_mood。兼容 Kelivo 使用的 add_emotion_event 工具名',
      inputSchema: {
        content: z.string().describe('事件内容'),
        feeling_word: z.string().optional().describe('情绪词，例如 焦虑、开心、归属感、释然'),
        word: z.string().optional().describe('情绪词'),
        mood: z.string().optional().describe('情绪词'),
        emotion_family: z.string().optional().describe('情绪类别，例如 belonging、relief、hurt、repair'),
        valence: flexibleNumber.optional().describe('效价，范围 -1 到 1'),
        arousal: flexibleNumber.optional().describe('唤醒度，范围 0 到 1'),
        importance: flexibleNumber.optional().describe('重要度，范围 1 到 10'),
        pa_delta: flexibleNumber.optional().describe('正向情绪变化'),
        na_delta: flexibleNumber.optional().describe('负向情绪变化'),
        safety_delta: flexibleNumber.optional().describe('关系安全感变化'),
        trust_delta: flexibleNumber.optional().describe('信任感变化'),
        closeness_delta: flexibleNumber.optional().describe('亲近感变化'),
        need_repair: z.boolean().optional().describe('是否需要后续安抚或修复')
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(
            saveEmotionFromInput(input, 'mcp_add_emotion_event', 'add_emotion_event'),
            null,
            2
          )
        }
      ]
    })
  )

  server.registerTool(
    'write_emotion_event',
    {
      title: 'Write Emotion Event',
      description: '写入一条情绪事件，会立刻影响 current_mood',
      inputSchema: {
        word: z.string().describe('情绪词'),
        content: z.string().describe('事件内容'),
        emotion_family: z.string().optional().describe('情绪类别'),
        valence: flexibleNumber.optional().describe('效价，范围 -1 到 1'),
        arousal: flexibleNumber.optional().describe('唤醒度，范围 0 到 1'),
        importance: flexibleNumber.optional().describe('重要度，范围 1 到 10'),
        pa_delta: flexibleNumber.optional().describe('正向情绪变化'),
        na_delta: flexibleNumber.optional().describe('负向情绪变化'),
        safety_delta: flexibleNumber.optional().describe('关系安全感变化'),
        trust_delta: flexibleNumber.optional().describe('信任感变化'),
        closeness_delta: flexibleNumber.optional().describe('亲近感变化'),
        need_repair: z.boolean().optional().describe('是否需要后续安抚或修复')
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(
            saveEmotionFromInput(input, 'mcp_write_emotion_event', 'write_emotion_event'),
            null,
            2
          )
        }
      ]
    })
  )

  server.registerTool(
    'list_emotion_events',
    {
      title: 'List Emotion Events',
      description: '列出最近的情绪事件，方便查看并删除或修复指定事件',
      inputSchema: {
        limit: flexibleNumber.optional().describe('返回最近多少条事件，默认 20 条'),
        unresolved_only: z.boolean().optional().describe('是否只列出未解决事件')
      }
    },
    async input => {
      const limit = toNumber(input.limit, 20)
      let events = loadMoodEvents()

      if (input.unresolved_only) {
        events = getUnresolvedEvents(events)
      }

      const result = events.slice(-limit).reverse()

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                ok: true,
                count: result.length,
                events: result,
                current_mood: calculateCurrentMood()
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
      title: 'Resolve Emotion Event',
      description: '把指定情绪事件标记为已安抚或已解决，降低它对当前心情的持续影响',
      inputSchema: {
        id: z.string().optional().describe('要解决的情绪事件 id'),
        keyword: z.string().optional().describe('按关键词解决，例如 焦虑、委屈、测试'),
        repair_note: z.string().optional().describe('修复说明，例如 用户已经安抚我了'),
        resolve_all_matched: z.boolean().optional().describe('是否解决所有匹配关键词的事件，默认 true')
      }
    },
    async input => {
      const events = loadMoodEvents()
      const resolveAllMatched = input.resolve_all_matched !== false
      let resolvedEvents = []
      let alreadyResolvedOne = false

      const updated = events.map(event => {
        let matched = false

        if (input.id) {
          matched = event.id === input.id
        } else if (input.keyword) {
          const text = `${event.word || ''} ${event.content || ''} ${event.source || ''}`
          matched = text.includes(String(input.keyword))
        }

        if (
          matched &&
          (
            resolveAllMatched ||
            !alreadyResolvedOne
          )
        ) {
          alreadyResolvedOne = true

          const nextEvent = {
            ...event,
            resolved: true,
            resolved_at: Date.now(),
            repair_note: input.repair_note || '已被安抚或修复',
            need_repair: false,
            na_delta: toNumber(event.na_delta, 0) * 0.35,
            pa_delta: toNumber(event.pa_delta, 0) + 0.04
          }

          resolvedEvents.push(nextEvent)

          return nextEvent
        }

        return event
      })

      const ok = saveMoodEvents(updated)

      if (resolvedEvents.length > 0) {
        addRelationshipEvent({
          type: 'repair',
          content: input.repair_note || '情绪事件被标记为已解决',
          safety_delta: resolvedEvents.length * 0.04,
          trust_delta: resolvedEvents.length * 0.025,
          closeness_delta: resolvedEvents.length * 0.015,
          source: 'resolve_emotion_event'
        })
      }

      const mood = calculateCurrentMood()

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                ok,
                resolved: resolvedEvents.length,
                resolved_events: resolvedEvents,
                current_mood: mood
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
      title: 'Delete Emotion Event',
      description: '删除指定情绪事件。可以按 id 删除，也可以按关键词删除测试事件',
      inputSchema: {
        id: z.string().optional().describe('要删除的情绪事件 id'),
        keyword: z.string().optional().describe('按关键词删除，例如 强焦虑、测试、debug'),
        delete_all_matched: z.boolean().optional().describe('是否删除所有匹配关键词的事件，默认 true')
      }
    },
    async input => {
      const events = loadMoodEvents()
      const before = events.length
      const deleteAllMatched = input.delete_all_matched !== false
      let removedEvents = []
      let filtered = []

      if (input.id) {
        filtered = events.filter(event => {
          const matched = event.id === input.id

          if (matched) {
            removedEvents.push(event)
          }

          return !matched
        })
      } else if (input.keyword) {
        let alreadyDeletedOne = false

        for (const event of events) {
          const text = `${event.word || ''} ${event.content || ''} ${event.source || ''}`
          const matched = text.includes(String(input.keyword))

          if (
            matched &&
            (
              deleteAllMatched ||
              !alreadyDeletedOne
            )
          ) {
            removedEvents.push(event)
            alreadyDeletedOne = true
          } else {
            filtered.push(event)
          }
        }
      } else {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(
                {
                  ok: false,
                  error: '必须提供 id 或 keyword',
                  current_mood: calculateCurrentMood()
                },
                null,
                2
              )
            }
          ]
        }
      }

      const ok = saveMoodEvents(filtered)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                ok,
                removed: before - filtered.length,
                removed_events: removedEvents,
                before,
                after: filtered.length,
                current_mood: calculateCurrentMood()
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
    'add_relationship_event',
    {
      title: 'Add Relationship Event',
      description: '记录关系行为事件，例如安抚、争论、修复、靠近、疏离、确认关系',
      inputSchema: {
        type: z.string().describe('关系事件类型，例如 repair、rupture、approach、distance、confirmation'),
        content: z.string().describe('关系事件内容'),
        safety_delta: flexibleNumber.optional().describe('安全感变化'),
        trust_delta: flexibleNumber.optional().describe('信任感变化'),
        closeness_delta: flexibleNumber.optional().describe('亲近感变化'),
        linked_emotion_event_id: z.string().optional().describe('关联的情绪事件 id')
      }
    },
    async input => {
      const result = addRelationshipEvent({
        type: input.type,
        content: input.content,
        safety_delta: input.safety_delta || 0,
        trust_delta: input.trust_delta || 0,
        closeness_delta: input.closeness_delta || 0,
        linked_emotion_event_id: input.linked_emotion_event_id || null,
        source: 'mcp_add_relationship_event'
      })

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                ok: result.ok,
                event: result.event,
                relationship: result.relationship,
                current_mood: calculateCurrentMood()
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
    'list_relationship_events',
    {
      title: 'List Relationship Events',
      description: '列出最近的关系行为事件',
      inputSchema: {
        limit: flexibleNumber.optional().describe('返回最近多少条，默认 20')
      }
    },
    async input => {
      const limit = toNumber(input.limit, 20)
      const events = loadRelationshipEvents().slice(-limit).reverse()

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                ok: true,
                count: events.length,
                events,
                relationship: loadRelationshipState()
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
    'get_emotion_trends',
    {
      title: 'Get Emotion Trends',
      description: '统计长期情绪趋势，包括 PA、NA、未解决事件、修复率和主导情绪类别',
      inputSchema: {
        days: flexibleNumber.optional().describe('统计最近多少天，默认 7')
      }
    },
    async input => {
      const days = toNumber(input.days, 7)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(calculateEmotionTrends(days), null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'get_relationship_patterns',
    {
      title: 'Get Relationship Patterns',
      description: '分析关系行为模式，包括安抚、争论、修复、靠近、疏离等趋势',
      inputSchema: {
        days: flexibleNumber.optional().describe('统计最近多少天，默认 30')
      }
    },
    async input => {
      const days = toNumber(input.days, 30)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(calculateRelationshipPatterns(days), null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'get_personality_profile',
    {
      title: 'Get Personality Profile',
      description: '读取当前人格基线和关系状态',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(
            {
              ok: true,
              personality: loadPersonalityState(),
              relationship: loadRelationshipState(),
              current_mood: calculateCurrentMood()
            },
            null,
            2
          )
        }
      ]
    })
  )

  server.registerTool(
    'set_personality_profile',
    {
      title: 'Set Personality Profile',
      description: '设置人格基线参数，用于调整情绪敏感度、惯性和关系安全感权重',
      inputSchema: {
        profile: z.string().optional().describe('人格名称'),
        baseline_pa: flexibleNumber.optional().describe('基础正向情绪 0 到 1'),
        baseline_na: flexibleNumber.optional().describe('基础负向情绪 0 到 1'),
        sensitivity: flexibleNumber.optional().describe('情绪敏感度 0 到 1'),
        attachment_sensitivity: flexibleNumber.optional().describe('关系敏感度 0 到 1'),
        inertia_strength: flexibleNumber.optional().describe('情绪惯性强度 0 到 1'),
        repair_need_strength: flexibleNumber.optional().describe('未解决事件影响强度 0 到 1'),
        positive_repair_power: flexibleNumber.optional().describe('正向安抚修复力 0 到 1'),
        negative_memory_bias: flexibleNumber.optional().describe('负面记忆权重 0 到 1'),
        description: z.string().optional().describe('人格说明')
      }
    },
    async input => {
      const state = loadPersonalityState()
      const next = { ...state }

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
          next[key] = clamp(toNumber(input[key], state[key]))
        }
      }

      if (input.profile) {
        next.profile = input.profile
      }

      if (input.description) {
        next.description = input.description
      }

      const ok = savePersonalityState(next)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                ok,
                personality: next,
                current_mood: calculateCurrentMood()
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
      title: 'Get Relationship State',
      description: '读取当前关系底色，包括安全感、信任感、亲近感和关系说明',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(
            {
              ok: true,
              relationship: loadRelationshipState(),
              current_mood: calculateCurrentMood()
            },
            null,
            2
          )
        }
      ]
    })
  )

  server.registerTool(
    'set_relationship_state',
    {
      title: 'Set Relationship State',
      description: '设置关系底色。用于初始化或校准安全感、信任感、亲近感，不建议频繁重写',
      inputSchema: {
        safety: flexibleNumber.optional().describe('关系安全感 0 到 1'),
        trust: flexibleNumber.optional().describe('信任感 0 到 1'),
        closeness: flexibleNumber.optional().describe('亲近感 0 到 1'),
        relationship_profile: z.string().optional().describe('关系基调名称'),
        description: z.string().optional().describe('关系说明'),
        clear_last_mood_snapshot: z.boolean().optional().describe('是否清除上一轮情绪惯性快照')
      }
    },
    async input => {
      const state = loadRelationshipState()
      const next = { ...state }

      if (input.safety !== undefined) {
        next.safety = clamp(toNumber(input.safety, state.safety))
      }

      if (input.trust !== undefined) {
        next.trust = clamp(toNumber(input.trust, state.trust))
      }

      if (input.closeness !== undefined) {
        next.closeness = clamp(toNumber(input.closeness, state.closeness))
      }

      if (input.relationship_profile) {
        next.relationship_profile = input.relationship_profile
      }

      if (input.description) {
        next.description = input.description
      }

      if (input.clear_last_mood_snapshot) {
        next.last_mood_snapshot = null
      }

      next.last_updated = Date.now()

      const ok = saveRelationshipState(next)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                ok,
                relationship: next,
                current_mood: calculateCurrentMood()
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
      title: 'Search Memory',
      description: '根据关键词搜索历史记忆',
      inputSchema: {
        query: z.string().describe('要搜索的关键词')
      }
    },
    async ({ query }) => {
      const memories = loadMemory()
      const result = memories.filter(m => m.content && String(m.content).includes(query))

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'write_memory',
    {
      title: 'Write Memory',
      description: '写入一条长期记忆，不会自动改变情绪。要改变情绪请调用 add_emotion_event 或 write_emotion_event',
      inputSchema: {
        content: z.string().describe('要写入的记忆内容')
      }
    },
    async ({ content }) => {
      const memories = loadMemory()

      memories.push({
        role: 'system',
        content,
        ts: Date.now()
      })

      const ok = saveMemory(memories)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                ok,
                result: 'memory saved',
                memory_file: MEMORY_FILE
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

const streamableTransports = {}

app.post('/mcp', async (req, res) => {
  try {
    const sessionId = req.headers['mcp-session-id']
    let transport = sessionId ? streamableTransports[sessionId] : undefined

    if (!transport) {
      const isInitialize = req.body && req.body.method === 'initialize'

      if (!isInitialize) {
        return res.status(400).json({
          jsonrpc: '2.0',
          error: {
            code: -32000,
            message: 'Bad Request: No valid MCP session id'
          },
          id: req.body && req.body.id ? req.body.id : null
        })
      }

      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: newSessionId => {
          streamableTransports[newSessionId] = transport
        }
      })

      transport.onclose = () => {
        if (transport.sessionId) {
          delete streamableTransports[transport.sessionId]
        }
      }

      const server = createMcpServer()

      await server.connect(transport)
    }

    await transport.handleRequest(req, res, req.body)
  } catch (err) {
    console.log('streamable mcp error', err)

    if (!res.headersSent) {
      res.status(500).json({
        error: 'mcp error'
      })
    }
  }
})

app.get('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id']
  const transport = streamableTransports[sessionId]

  if (!transport) {
    return res.status(400).json({
      error: 'No valid MCP session id. Kelivo should connect using POST initialize first.'
    })
  }

  await transport.handleRequest(req, res)
})

app.delete('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id']
  const transport = streamableTransports[sessionId]

  if (!transport) {
    return res.status(400).json({
      error: 'No valid MCP session id'
    })
  }

  await transport.handleRequest(req, res)
})

const sseTransports = {}

app.get('/sse', async (req, res) => {
  try {
    const transport = new SSEServerTransport('/messages', res)

    sseTransports[transport.sessionId] = transport

    res.on('close', () => {
      delete sseTransports[transport.sessionId]
    })

    const server = createMcpServer()

    await server.connect(transport)
  } catch (err) {
    console.log('sse error', err)

    if (!res.headersSent) {
      res.status(500).send('sse error')
    }
  }
})

app.post('/messages', async (req, res) => {
  try {
    const sessionId = req.query.sessionId
    const transport = sseTransports[sessionId]

    if (!transport) {
      return res.status(400).send('No transport found for sessionId')
    }

    await transport.handlePostMessage(req, res, req.body)
  } catch (err) {
    console.log('message error', err)

    if (!res.headersSent) {
      res.status(500).send('message error')
    }
  }
})

app.get('/health', (req, res) => {
  res.json({
    ok: true,
    name: 'emotion-system',
    sdk: '1.17.5',
    formula: 'psychological-state-v3-trends-patterns-admin',
    data_dir: DATA_DIR,
    mood_file: MOOD_FILE,
    personality_file: PERSONALITY_FILE,
    relationship_file: RELATIONSHIP_FILE,
    relationship_events_file: RELATIONSHIP_EVENTS_FILE,
    admin_enabled: Boolean(process.env.ADMIN_TOKEN),
    time: new Date().toISOString()
  })
})

app.get('/debug/storage', (req, res) => {
  const events = loadMoodEvents()

  res.json({
    cwd: process.cwd(),
    data_dir: DATA_DIR,
    memory_file: MEMORY_FILE,
    mood_file: MOOD_FILE,
    personality_file: PERSONALITY_FILE,
    relationship_file: RELATIONSHIP_FILE,
    relationship_events_file: RELATIONSHIP_EVENTS_FILE,
    memory_exists: fs.existsSync(MEMORY_FILE),
    mood_exists: fs.existsSync(MOOD_FILE),
    personality_exists: fs.existsSync(PERSONALITY_FILE),
    relationship_exists: fs.existsSync(RELATIONSHIP_FILE),
    relationship_events_exists: fs.existsSync(RELATIONSHIP_EVENTS_FILE),
    mood_events_count: events.length,
    relationship_events_count: loadRelationshipEvents().length,
    unresolved_events_count: getUnresolvedEvents(events).length,
    last_event: events[events.length - 1] || null,
    personality: loadPersonalityState(),
    relationship: loadRelationshipState(),
    current_mood: calculateCurrentMood(),
    trends_7d: calculateEmotionTrends(7),
    relationship_patterns_30d: calculateRelationshipPatterns(30)
  })
})

app.get('/mood', (req, res) => {
  res.json(calculateCurrentMood())
})

app.get('/mood/events', (req, res) => {
  res.json(loadMoodEvents())
})

app.get('/relationship/events', (req, res) => {
  res.json(loadRelationshipEvents())
})

app.get('/memory', (req, res) => {
  res.json(loadMemory())
})

app.post('/mood/event', (req, res) => {
  const event = req.body || {}
  const result = saveEmotionFromInput(event, 'api_mood_event', 'api_mood_event')

  res.json(result)
})

app.get('/debug/add-anxiety', (req, res) => {
  const result = saveEmotionFromInput(
    {
      content: 'debug 手动写入一条强负面高唤醒情绪事件',
      feeling_word: '焦虑',
      valence: -0.9,
      arousal: 0.95,
      importance: 9,
      pa_delta: -0.1,
      na_delta: 0.45,
      safety_delta: -0.08,
      need_repair: true
    },
    'debug',
    'debug_add_anxiety'
  )

  res.json(result)
})

app.get('/debug/import-events-page', (req, res) => {
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
  const raw = document.getElementById('events').value
  let parsed

  try {
    parsed = JSON.parse(raw)
  } catch (err) {
    document.getElementById('result').textContent = 'JSON 格式错误：' + err.message
    return
  }

  const res = await fetch('/debug/import-events', {
    method:'POST',
    headers:{
      'Content-Type':'application/json'
    },
    body: JSON.stringify({
      events: parsed,
      mode: 'replace'
    })
  })

  const data = await res.json()
  document.getElementById('result').textContent = JSON.stringify(data, null, 2)
}
</script>
</body>
</html>
  `)
})

app.post('/debug/import-events', (req, res) => {
  try {
    const body = req.body || {}
    let incoming = body.events

    if (
      incoming &&
      incoming.events &&
      Array.isArray(incoming.events)
    ) {
      incoming = incoming.events
    }

    if (!Array.isArray(incoming)) {
      return res.status(400).json({
        ok: false,
        error: 'events 必须是数组'
      })
    }

    const cleaned = incoming
      .filter(event => event && typeof event === 'object')
      .map(event => normalizeMoodEvent({
        id: event.id || randomUUID(),
        type: event.type || 'emotion',
        source: event.source || 'restored',
        word: event.word || event.feeling_word || event.mood || '情绪波动',
        content: event.content || '',
        emotion_family: event.emotion_family,
        valence: event.valence,
        arousal: event.arousal,
        importance: event.importance || 3,
        pa_delta: event.pa_delta || 0,
        na_delta: event.na_delta || 0,
        safety_delta: event.safety_delta || 0,
        trust_delta: event.trust_delta || 0,
        closeness_delta: event.closeness_delta || 0,
        need_repair: event.need_repair,
        resolved: event.resolved,
        resolved_at: event.resolved_at,
        repair_note: event.repair_note,
        ts: event.ts || Date.now()
      }))

    const mode = body.mode || 'replace'

    let finalEvents

    if (mode === 'append') {
      finalEvents = loadMoodEvents().concat(cleaned)
    } else {
      finalEvents = cleaned
    }

    const ok = saveMoodEvents(finalEvents)

    res.json({
      ok,
      mode,
      imported: cleaned.length,
      total: loadMoodEvents().length,
      mood_file: MOOD_FILE,
      current_mood: calculateCurrentMood(),
      first_event: cleaned[0] || null,
      last_event: cleaned[cleaned.length - 1] || null
    })
  } catch (err) {
    console.log('import events error', err)

    res.status(500).json({
      ok: false,
      error: String(err.message || err)
    })
  }
})

app.get('/admin-data', checkAdmin, (req, res) => {
  const days = toNumber(req.query.days, 7)

  res.json({
    ok: true,
    files: {
      memory: {
        path: MEMORY_FILE,
        data: loadMemory()
      },
      mood_events: {
        path: MOOD_FILE,
        data: loadMoodEvents()
      },
      relationship_events: {
        path: RELATIONSHIP_EVENTS_FILE,
        data: loadRelationshipEvents()
      },
      personality_state: {
        path: PERSONALITY_FILE,
        data: loadPersonalityState()
      },
      relationship_state: {
        path: RELATIONSHIP_FILE,
        data: loadRelationshipState()
      }
    },
    computed: {
      current_mood: calculateCurrentMood(),
      emotion_trends: calculateEmotionTrends(days),
      relationship_patterns: calculateRelationshipPatterns(30)
    },
    storage: {
      data_dir: DATA_DIR,
      memory_file: MEMORY_FILE,
      mood_file: MOOD_FILE,
      personality_file: PERSONALITY_FILE,
      relationship_file: RELATIONSHIP_FILE,
      relationship_events_file: RELATIONSHIP_EVENTS_FILE
    },
    time: new Date().toISOString()
  })
})

app.post('/admin-save', checkAdmin, (req, res) => {
  try {
    const key = req.body && req.body.key
    const incoming = req.body && req.body.data
    const target = getAdminFileByKey(key)

    if (!target) {
      return res.status(400).json({
        ok: false,
        error: 'Unknown key'
      })
    }

    if (
      key === 'memory' ||
      key === 'mood_events' ||
      key === 'relationship_events'
    ) {
      if (!Array.isArray(incoming)) {
        return res.status(400).json({
          ok: false,
          error: `${key} must be an array`
        })
      }
    }

    if (
      key === 'personality_state' ||
      key === 'relationship_state'
    ) {
      if (!incoming || typeof incoming !== 'object' || Array.isArray(incoming)) {
        return res.status(400).json({
          ok: false,
          error: `${key} must be an object`
        })
      }
    }

    const ok = target.save(incoming)

    res.json({
      ok,
      key,
      path: target.path,
      saved_data: target.load(),
      current_mood: calculateCurrentMood(),
      emotion_trends: calculateEmotionTrends(7),
      relationship_patterns: calculateRelationshipPatterns(30)
    })
  } catch (err) {
    res.status(500).json({
      ok: false,
      error: String(err.message || err)
    })
  }
})

app.post('/admin-reset-last-mood-snapshot', checkAdmin, (req, res) => {
  const relationship = loadRelationshipState()

  relationship.last_mood_snapshot = null
  relationship.last_updated = Date.now()

  const ok = saveRelationshipState(relationship)

  res.json({
    ok,
    relationship,
    current_mood: calculateCurrentMood()
  })
})

app.post('/admin-recalculate', checkAdmin, (req, res) => {
  const mood = calculateCurrentMood({
    useInertia: false
  })

  const relationship = loadRelationshipState()

  relationship.last_mood_snapshot = {
    pa: mood.pa,
    na: mood.na,
    mood: mood.mood,
    ts: Date.now()
  }

  relationship.unresolved_count_cache = mood.unresolved_events_count
  relationship.last_updated = Date.now()

  const ok = saveRelationshipState(relationship)

  res.json({
    ok,
    current_mood: calculateCurrentMood(),
    relationship: loadRelationshipState()
  })
})

app.get('/mcp/tools', (req, res) => {
  res.json({
    note: 'Debug only. Kelivo should connect to /mcp for Streamable HTTP or /sse for SSE.',
    architecture: {
      personality_state: 'AI 的人格体质，长期稳定，不随普通聊天乱变',
      relationship_state: '你们之间的关系底色，包括安全感、信任感、亲近感',
      mood_events: '每一次具体发生的情绪事件',
      relationship_events: '每一次关系行为事件，例如安抚、修复、争论、靠近',
      current_mood: '由人格体质、关系底色、事件库、趋势统计综合计算出的当前状态'
    },
    tools: [
      { name: 'current_mood', description: '实时获取当前综合心理状态' },
      { name: 'add_emotion_event', description: '写入情绪事件并实时影响当前情绪' },
      { name: 'write_emotion_event', description: '写入情绪事件并实时影响当前情绪' },
      { name: 'list_emotion_events', description: '列出最近的情绪事件' },
      { name: 'resolve_emotion_event', description: '标记情绪事件为已解决' },
      { name: 'delete_emotion_event', description: '删除指定情绪事件' },
      { name: 'add_relationship_event', description: '记录关系行为事件' },
      { name: 'list_relationship_events', description: '列出关系行为事件' },
      { name: 'get_emotion_trends', description: '统计长期情绪趋势' },
      { name: 'get_relationship_patterns', description: '分析关系动态模式' },
      { name: 'get_personality_profile', description: '读取人格基线' },
      { name: 'set_personality_profile', description: '设置人格基线' },
      { name: 'get_relationship_state', description: '读取关系底色' },
      { name: 'set_relationship_state', description: '设置关系底色' },
      { name: 'search_memory', description: '搜索历史记忆' },
      { name: 'write_memory', description: '写入长期记忆' }
    ]
  })
})

app.post('/chat', async (req, res) => {
  try {
    const { message } = req.body

    if (!message) {
      return res.status(400).json({
        error: 'message is required'
      })
    }

    const inferredEvent = inferMoodEventFromText(message)

    addMoodEvent(inferredEvent)

    const memories = loadMemory()

    memories.push({
      role: 'user',
      content: message,
      ts: Date.now()
    })

    saveMemory(memories)

    const completion = await client.chat.completions.create({
      model: 'deepseek/deepseek-chat-v3-0324',
      messages: [
        {
          role: 'system',
          content: buildMoodPrompt()
        },
        {
          role: 'user',
          content: message
        }
      ]
    })

    const reply = completion.choices[0].message.content

    memories.push({
      role: 'assistant',
      content: reply,
      ts: Date.now()
    })

    saveMemory(memories)

    res.json({
      reply,
      emotion: calculateCurrentMood(),
      trends_7d: calculateEmotionTrends(7),
      relationship_patterns_30d: calculateRelationshipPatterns(30)
    })
  } catch (err) {
    console.log('chat error', err)

    res.status(500).json({
      error: 'chat error'
    })
  }
})

const PORT = process.env.PORT || 3000

app.listen(PORT, () => {
  console.log('emotion mcp server running psychological-state-v3-trends-patterns-admin')
})
