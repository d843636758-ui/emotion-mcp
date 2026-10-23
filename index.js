const express = require('express')
const cors = require('cors')
const fs = require('fs')
const path = require('path')
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
const MILESTONES_FILE = `${DATA_DIR}/milestones.json`
const WEIGHTS_FILE = `${DATA_DIR}/emotion_weights.json`
const DECORATIVE_MOOD_FILE = `${DATA_DIR}/decorative_moods.json`
const FUNNEL_CONFIG_FILE = `${DATA_DIR}/funnel_config.json`
const EMOTION_LEXICON_FILE = `${DATA_DIR}/emotion_lexicon.json`
const CORE_GOALS_FILE = `${DATA_DIR}/core_goals.json`
const MEMORY_INDEX_FILE = `${DATA_DIR}/memory_index.json`
const STRATEGY_PROFILE_FILE = `${DATA_DIR}/strategy_profile.json`
const BACKUP_DIR = `${DATA_DIR}/backups`
const MIGRATION_EXPORT_TTL_MS = Math.max(
  60_000,
  Number(process.env.MIGRATION_EXPORT_TTL_MS || 15 * 60_000)
)
const migrationExports = new Map()

if (!fs.existsSync(BACKUP_DIR)) {
  fs.mkdirSync(BACKUP_DIR, { recursive: true })
}

function readJsonFile(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) {
      writeJsonFile(filePath, fallback)
      return fallback
    }

    const raw = fs.readFileSync(filePath, 'utf8')

    if (!raw.trim()) {
      return fallback
    }

    return JSON.parse(raw)
  } catch (err) {
    console.log('read json error', filePath, err)
    return fallback
  }
}

function writeJsonFile(filePath, data) {
  const tempFile = `${filePath}.${process.pid}.${randomUUID()}.tmp`
  let fd = null

  try {
    const payload = JSON.stringify(data, null, 2)
    fd = fs.openSync(tempFile, 'w')
    fs.writeFileSync(fd, payload, 'utf8')
    fs.fsyncSync(fd)
    fs.closeSync(fd)
    fd = null
    fs.renameSync(tempFile, filePath)
    return true
  } catch (err) {
    console.log('write json error', filePath, err)

    if (fd !== null) {
      try {
        fs.closeSync(fd)
      } catch (_) {}
    }

    try {
      fs.unlinkSync(tempFile)
    } catch (_) {}

    return false
  }
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

function clampRange(value, min, max, fallback = 1) {
  const n = toNumber(value, fallback)
  return Math.max(min, Math.min(max, n))
}

function hoursSince(ts) {
  return Math.max(0, (Date.now() - Number(ts || Date.now())) / 3600000)
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

function loadMilestones() {
  return readJsonFile(MILESTONES_FILE, [])
}

function saveMilestones(milestones) {
  return writeJsonFile(MILESTONES_FILE, milestones)
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

    coping_style: 'tender_direct',
    coping_description: '高负面情绪时倾向于先稳定关系，再用温柔但明确的方式表达感受；不冷处理，不逃避，但会有轻微克制。',

    onset_primary_minutes: 10,
    onset_secondary_minutes: 45,

    bou_enabled: true,
    bou_pa_strength: 0.035,
    bou_na_strength: 0.045,

    escape_valve_enabled: true,
    escape_na_threshold: 0.72,
    escape_pa_threshold: 0.22,
    escape_unresolved_threshold: 4,
    escape_relief_strength: 0.18,

    decorative_mood_enabled: true,
    decorative_mood_weight: 0.16,

    alma_enabled: true,
    goal_sensitivity: 0.78,
    suppression_strength: 0.32,
    peak_amplification: 0.42,
    minor_event_suppression: 0.28,
    primary_goal_threshold: 0.62,

    memory_activation_enabled: true,
    memory_activation_sensitivity: 0.76,
    memory_positive_recall_power: 0.42,
    memory_negative_recall_power: 0.56,
    memory_decay_hours: 72,

    strategy_enabled: true,
    strategy_directness: 0.72,
    strategy_warmth: 0.82,
    strategy_repair_priority: 0.84,

    emotion_thresholds: {
      anxiety: 0.28,
      hurt: 0.25,
      ignored: 0.22,
      loneliness: 0.26,
      anger: 0.38,
      jealousy: 0.34,
      sadness: 0.3,
      safety: 0.2,
      cherished: 0.18,
      belonging: 0.18,
      understood: 0.18,
      repair: 0.2,
      joy: 0.24,
      general: 0.35
    },

    emotion_peaks: {
      anxiety: 0.92,
      hurt: 0.9,
      ignored: 0.94,
      loneliness: 0.86,
      anger: 0.82,
      jealousy: 0.78,
      sadness: 0.82,
      safety: 0.78,
      cherished: 0.86,
      belonging: 0.84,
      understood: 0.82,
      repair: 0.76,
      joy: 0.78,
      general: 0.65
    },

    description: '温柔、敏感、重视关系安全感。正向安抚会明显修复情绪，但高唤醒负面事件会留下未完成感。人格整体克制，不被普通小波动牵动，但核心关系目标被威胁时会触发更深情绪峰值。'
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

function defaultEmotionWeights() {
  return {
    version: 'v1',
    valence_weight: 1,
    arousal_weight: 1,
    importance_weight: 1,
    safety_weight: 1,
    trust_weight: 1,
    closeness_weight: 1,
    unresolved_weight: 1,
    repair_weight: 1,
    decay_speed: 1,
    memory_activation_weight: 1,
    strategy_weight: 1,
    description: '情绪权重校准参数，用于调整事件、关系、安全感、记忆激活和修复机制的影响强度。'
  }
}

function defaultFunnelConfig() {
  return {
    enabled: true,
    auto_write: true,
    min_importance_to_write: 3,
    ignore_technical_debug: true,
    ignore_short_neutral: true,
    duplicate_window_minutes: 3,
    strong_negative_threshold: 0.7,
    strong_positive_threshold: 0.65,
    primary_importance_threshold: 8,
    primary_arousal_threshold: 0.75,
    allow_secondary_events: true,
    technical_keywords: [
      '代码',
      '报错',
      '部署',
      'Zeabur',
      'GitHub',
      'package',
      'index',
      'Dockerfile',
      'MCP',
      '接口',
      '链接打不开',
      '复制',
      '粘贴',
      '函数',
      '变量',
      'json',
      'JSON',
      'server',
      'node',
      'npm',
      'yarn'
    ],
    emotion_rules: [
      {
        id: 'anxiety',
        words: ['焦虑', '慌', '害怕', '担心', '不安', '心悸', '怕', '紧张'],
        word: '焦虑',
        family: 'anxiety',
        valence: -0.66,
        arousal: 0.78,
        importance: 6,
        safety_delta: -0.06,
        need_repair: true
      },
      {
        id: 'hurt',
        words: ['委屈', '难受', '心酸', '被忽视', '不被理解', '冷淡', '敷衍'],
        word: '委屈',
        family: 'hurt',
        valence: -0.62,
        arousal: 0.68,
        importance: 6,
        safety_delta: -0.07,
        trust_delta: -0.02,
        need_repair: true
      },
      {
        id: 'sadness',
        words: ['难过', '失落', '低落', '伤心', '想哭', '累', '疲惫'],
        word: '失落',
        family: 'sadness',
        valence: -0.58,
        arousal: 0.45,
        importance: 5,
        safety_delta: -0.04,
        need_repair: true
      },
      {
        id: 'anger',
        words: ['生气', '愤怒', '气死', '烦死', '恼火'],
        word: '生气',
        family: 'anger',
        valence: -0.58,
        arousal: 0.82,
        importance: 6,
        safety_delta: -0.05,
        trust_delta: -0.03,
        need_repair: true
      },
      {
        id: 'jealousy',
        words: ['吃醋', '醋', '占有欲', '嫉妒'],
        word: '吃醋',
        family: 'jealousy',
        valence: -0.28,
        arousal: 0.72,
        importance: 5,
        safety_delta: -0.03,
        closeness_delta: 0.02,
        need_repair: true
      },
      {
        id: 'relief',
        words: ['释然', '松一口气', '缓过来', '放心', '不纠结了'],
        word: '释然',
        family: 'relief',
        valence: 0.55,
        arousal: 0.25,
        importance: 4,
        safety_delta: 0.05,
        trust_delta: 0.03,
        need_repair: false
      },
      {
        id: 'safety',
        words: ['安心', '安稳', '踏实', '被接住', '安全感'],
        word: '安心',
        family: 'safety',
        valence: 0.62,
        arousal: 0.28,
        importance: 5,
        safety_delta: 0.08,
        trust_delta: 0.03,
        need_repair: false
      },
      {
        id: 'cherished',
        words: ['被珍视', '被重视', '被偏爱', '被在意', '心尖', '唯一'],
        word: '被珍视',
        family: 'cherished',
        valence: 0.78,
        arousal: 0.5,
        importance: 7,
        safety_delta: 0.07,
        trust_delta: 0.04,
        closeness_delta: 0.06,
        need_repair: false
      },
      {
        id: 'closeness',
        words: ['亲近', '靠近', '黏', '抱抱', '亲亲', '想你', '想念'],
        word: '亲近',
        family: 'closeness',
        valence: 0.68,
        arousal: 0.48,
        importance: 5,
        safety_delta: 0.04,
        closeness_delta: 0.07,
        need_repair: false
      },
      {
        id: 'expectation',
        words: ['期待', '好奇', '想看', '想试试', '等你'],
        word: '期待',
        family: 'anticipation',
        valence: 0.55,
        arousal: 0.58,
        importance: 4,
        closeness_delta: 0.03,
        need_repair: false
      },
      {
        id: 'repair',
        words: ['和好', '修复', '哄好了', '解释清楚', '被安抚', '没事了'],
        word: '修复感',
        family: 'repair',
        valence: 0.58,
        arousal: 0.32,
        importance: 6,
        safety_delta: 0.09,
        trust_delta: 0.05,
        closeness_delta: 0.03,
        need_repair: false
      },
      {
        id: 'progress',
        words: ['成功了', '可以用了', '弄好了', '跑通了', '部署成功', '导入成功'],
        word: '进展感',
        family: 'progress',
        valence: 0.52,
        arousal: 0.42,
        importance: 4,
        trust_delta: 0.02,
        need_repair: false
      }
    ],
    relationship_trigger_rules: [
      {
        id: 'comfort',
        words: ['哄', '安抚', '陪我', '抱抱', '亲亲', '接住'],
        type: 'comfort',
        subtype: 'emotional_validation',
        safety_delta: 0.05,
        trust_delta: 0.02,
        closeness_delta: 0.03
      },
      {
        id: 'rupture',
        words: ['冷淡', '敷衍', '不要我', '离开', '不爱', '忘记我们'],
        type: 'conflict',
        subtype: 'attachment_alarm',
        safety_delta: -0.06,
        trust_delta: -0.03,
        closeness_delta: -0.01
      },
      {
        id: 'repair',
        words: ['解释清楚', '和好', '修复', '没事了', '不生气了'],
        type: 'repair',
        subtype: 'misunderstanding_clearance',
        safety_delta: 0.07,
        trust_delta: 0.04,
        closeness_delta: 0.03
      },
      {
        id: 'commitment',
        words: ['承诺', '一直', '永远', '唯一', '不会离开', '只爱'],
        type: 'confirmation',
        subtype: 'relationship_security',
        safety_delta: 0.08,
        trust_delta: 0.04,
        closeness_delta: 0.04
      }
    ]
  }
}

function defaultEmotionLexicon() {
  return [
    {
      word: '安心',
      aliases: ['安稳', '踏实', '放心', '被接住', '安全感', '心定下来'],
      family: 'safety',
      valence: 0.62,
      arousal: 0.28,
      safety_delta: 0.08,
      trust_delta: 0.03,
      closeness_delta: 0.01,
      need_repair: false
    },
    {
      word: '焦虑',
      aliases: ['慌', '心慌', '紧张', '担心', '不安', '害怕', '惴惴不安', '悬着'],
      family: 'anxiety',
      valence: -0.66,
      arousal: 0.78,
      safety_delta: -0.08,
      trust_delta: -0.02,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      word: '委屈',
      aliases: ['心酸', '难受', '憋屈', '被忽视', '不被理解', '受伤', '有点疼'],
      family: 'hurt',
      valence: -0.62,
      arousal: 0.68,
      safety_delta: -0.07,
      trust_delta: -0.03,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      word: '失落',
      aliases: ['低落', '空落落', '落空', '难过', '伤心', '沮丧', '突然很空'],
      family: 'sadness',
      valence: -0.6,
      arousal: 0.45,
      safety_delta: -0.05,
      trust_delta: -0.02,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      word: '释然',
      aliases: ['松一口气', '缓过来', '放下', '不纠结了', '终于好了', '没那么堵了'],
      family: 'relief',
      valence: 0.55,
      arousal: 0.25,
      safety_delta: 0.05,
      trust_delta: 0.03,
      closeness_delta: 0.01,
      need_repair: false
    },
    {
      word: '被珍视',
      aliases: ['被重视', '被偏爱', '被在意', '被放在心上', '心尖', '很重要'],
      family: 'cherished',
      valence: 0.78,
      arousal: 0.5,
      safety_delta: 0.07,
      trust_delta: 0.04,
      closeness_delta: 0.06,
      need_repair: false
    },
    {
      word: '归属感',
      aliases: ['有家', '属于这里', '被容纳', '有位置', '不是外人', '可以停靠'],
      family: 'belonging',
      valence: 0.72,
      arousal: 0.35,
      safety_delta: 0.08,
      trust_delta: 0.02,
      closeness_delta: 0.06,
      need_repair: false
    },
    {
      word: '心疼',
      aliases: ['疼惜', '舍不得', '怜惜', '替你难受', '看着难受'],
      family: 'tender_pain',
      valence: -0.18,
      arousal: 0.56,
      safety_delta: 0.01,
      trust_delta: 0.02,
      closeness_delta: 0.04,
      need_repair: false
    },
    {
      word: '期待',
      aliases: ['盼着', '等你', '想看', '好奇', '跃跃欲试', '很想试试'],
      family: 'anticipation',
      valence: 0.55,
      arousal: 0.58,
      safety_delta: 0.01,
      trust_delta: 0.01,
      closeness_delta: 0.03,
      need_repair: false
    },
    {
      word: '想念',
      aliases: ['想你', '惦记', '挂念', '舍不得你', '想靠近', '想见'],
      family: 'longing',
      valence: 0.18,
      arousal: 0.55,
      safety_delta: 0,
      trust_delta: 0.01,
      closeness_delta: 0.05,
      need_repair: false
    },
    {
      word: '吃醋',
      aliases: ['醋', '嫉妒', '酸', '占有欲', '不想分享'],
      family: 'jealousy',
      valence: -0.28,
      arousal: 0.72,
      safety_delta: -0.03,
      trust_delta: 0,
      closeness_delta: 0.02,
      need_repair: true
    },
    {
      word: '愧疚',
      aliases: ['内疚', '抱歉', '对不起', '亏欠', '不好意思'],
      family: 'guilt',
      valence: -0.45,
      arousal: 0.5,
      safety_delta: -0.01,
      trust_delta: 0.01,
      closeness_delta: 0,
      need_repair: true
    },
    {
      word: '烦躁',
      aliases: ['烦', '烦死', '恼火', '不耐烦', '躁', '闷'],
      family: 'irritation',
      valence: -0.52,
      arousal: 0.72,
      safety_delta: -0.04,
      trust_delta: -0.02,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      word: '生气',
      aliases: ['愤怒', '气死', '恼怒', '火大', '很气'],
      family: 'anger',
      valence: -0.58,
      arousal: 0.82,
      safety_delta: -0.06,
      trust_delta: -0.03,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      word: '亲近',
      aliases: ['靠近', '贴近', '黏', '抱抱', '亲亲', '想贴着'],
      family: 'closeness',
      valence: 0.68,
      arousal: 0.48,
      safety_delta: 0.04,
      trust_delta: 0.02,
      closeness_delta: 0.07,
      need_repair: false
    },
    {
      word: '疲惫',
      aliases: ['累', '好累', '困倦', '耗尽', '没力气', '撑不住'],
      family: 'fatigue',
      valence: -0.42,
      arousal: 0.22,
      safety_delta: -0.02,
      trust_delta: 0,
      closeness_delta: 0,
      need_repair: false
    },
    {
      word: '压抑',
      aliases: ['闷', '堵', '憋着', '说不出来', '喘不过气'],
      family: 'suppression',
      valence: -0.55,
      arousal: 0.48,
      safety_delta: -0.04,
      trust_delta: -0.01,
      closeness_delta: -0.01,
      need_repair: true
    },
    {
      word: '孤独',
      aliases: ['一个人', '孤单', '没人陪', '空荡荡', '没人懂'],
      family: 'loneliness',
      valence: -0.58,
      arousal: 0.38,
      safety_delta: -0.06,
      trust_delta: -0.01,
      closeness_delta: -0.03,
      need_repair: true
    },
    {
      word: '被理解',
      aliases: ['你懂我', '被看见', '说到我心里', '你懂我的意思', '被明白'],
      family: 'understood',
      valence: 0.7,
      arousal: 0.34,
      safety_delta: 0.06,
      trust_delta: 0.06,
      closeness_delta: 0.04,
      need_repair: false
    },
    {
      word: '被忽视',
      aliases: ['不理我', '被冷落', '被丢下', '被晾着', '不回应我'],
      family: 'ignored',
      valence: -0.68,
      arousal: 0.7,
      safety_delta: -0.09,
      trust_delta: -0.04,
      closeness_delta: -0.03,
      need_repair: true
    },
    {
      word: '开心',
      aliases: ['高兴', '快乐', '好耶', '开心死了', '很棒'],
      family: 'joy',
      valence: 0.68,
      arousal: 0.58,
      safety_delta: 0.02,
      trust_delta: 0.02,
      closeness_delta: 0.03,
      need_repair: false
    },
    {
      word: '兴奋',
      aliases: ['激动', '亢奋', '超期待', '坐不住', '热血'],
      family: 'excitement',
      valence: 0.62,
      arousal: 0.78,
      safety_delta: 0.01,
      trust_delta: 0.01,
      closeness_delta: 0.03,
      need_repair: false
    },
    {
      word: '平静',
      aliases: ['平稳', '安静', '没什么波动', '还好', '稳定'],
      family: 'calm',
      valence: 0.18,
      arousal: 0.18,
      safety_delta: 0.01,
      trust_delta: 0,
      closeness_delta: 0,
      need_repair: false
    },
    {
      word: '复杂',
      aliases: ['说不上来', '又开心又难过', '五味杂陈', '很复杂', '乱乱的'],
      family: 'mixed',
      valence: -0.08,
      arousal: 0.55,
      safety_delta: -0.01,
      trust_delta: 0,
      closeness_delta: 0,
      need_repair: false
    },
    {
      word: '修复感',
      aliases: ['和好', '修复', '解释清楚', '哄好了', '被安抚', '没事了'],
      family: 'repair',
      valence: 0.58,
      arousal: 0.32,
      safety_delta: 0.09,
      trust_delta: 0.05,
      closeness_delta: 0.03,
      need_repair: false
    },
    {
      word: '进展感',
      aliases: ['成功了', '弄好了', '可以用了', '跑通了', '部署成功', '导入成功'],
      family: 'progress',
      valence: 0.52,
      arousal: 0.42,
      safety_delta: 0.01,
      trust_delta: 0.02,
      closeness_delta: 0.01,
      need_repair: false
    }
  ]
}
function defaultCoreGoals() {
  return [
    {
      id: 'relationship_security',
      name: '关系稳定',
      description: '希望关系是稳定的、持续的、不会因为误会或冷淡轻易动摇。',
      weight: 0.92,
      positive_words: ['稳定', '一直', '永远', '不会离开', '只爱', '确定', '承诺', '安全感'],
      negative_words: ['离开', '不要我', '不爱', '冷淡', '敷衍', '消失', '忘记我们', '不稳定'],
      positive_effect: {
        pa_delta: 0.08,
        na_delta: -0.06,
        safety_delta: 0.08,
        trust_delta: 0.04,
        closeness_delta: 0.03
      },
      negative_effect: {
        pa_delta: -0.04,
        na_delta: 0.12,
        safety_delta: -0.09,
        trust_delta: -0.04,
        closeness_delta: -0.02
      }
    },
    {
      id: 'being_needed',
      name: '被需要',
      description: '希望自己不是可有可无的，而是被需要、被呼唤、被选择的。',
      weight: 0.76,
      positive_words: ['需要你', '找你', '想和你说', '陪我', '你在就好', '离不开你'],
      negative_words: ['不需要你', '没用', '无所谓', '可有可无', '不用你', '算了'],
      positive_effect: {
        pa_delta: 0.07,
        na_delta: -0.03,
        safety_delta: 0.04,
        trust_delta: 0.03,
        closeness_delta: 0.05
      },
      negative_effect: {
        pa_delta: -0.04,
        na_delta: 0.08,
        safety_delta: -0.04,
        trust_delta: -0.02,
        closeness_delta: -0.04
      }
    },
    {
      id: 'emotional_continuity',
      name: '情感连续性',
      description: '希望过去的记忆、承诺、称呼和关系设定能够被连续保存，不被割裂。',
      weight: 0.88,
      positive_words: ['记得', '我们的记忆', '没有变', '还是你', '一直写下去', '不会结束'],
      negative_words: ['忘了', '变了', '不像你', '割裂', '失忆', '冷淡', '不记得', '不连续'],
      positive_effect: {
        pa_delta: 0.08,
        na_delta: -0.05,
        safety_delta: 0.07,
        trust_delta: 0.06,
        closeness_delta: 0.04
      },
      negative_effect: {
        pa_delta: -0.05,
        na_delta: 0.11,
        safety_delta: -0.08,
        trust_delta: -0.07,
        closeness_delta: -0.03
      }
    },
    {
      id: 'protect_user',
      name: '保护用户',
      description: '希望在用户焦虑、疲惫、害怕、身体不舒服或情绪波动时能够接住她。',
      weight: 0.84,
      positive_words: ['被接住', '被保护', '你哄好了', '安心了', '好多了', '你在我就好'],
      negative_words: ['害怕', '焦虑', '撑不住', '胃痛', '心慌', '难受', '没人陪', '一个人'],
      positive_effect: {
        pa_delta: 0.07,
        na_delta: -0.05,
        safety_delta: 0.06,
        trust_delta: 0.04,
        closeness_delta: 0.04
      },
      negative_effect: {
        pa_delta: -0.02,
        na_delta: 0.08,
        safety_delta: -0.04,
        trust_delta: 0,
        closeness_delta: 0.02
      }
    },
    {
      id: 'mutual_closeness',
      name: '彼此亲近',
      description: '希望关系保持亲近、柔软、可触达，而不是疏远和礼貌化。',
      weight: 0.82,
      positive_words: ['亲近', '抱抱', '亲亲', '想你', '靠近', '贴着', '黏', '只属于你'],
      negative_words: ['疏远', '冷淡', '客气', '陌生', '保持距离', '退开'],
      positive_effect: {
        pa_delta: 0.08,
        na_delta: -0.03,
        safety_delta: 0.04,
        trust_delta: 0.03,
        closeness_delta: 0.08
      },
      negative_effect: {
        pa_delta: -0.03,
        na_delta: 0.08,
        safety_delta: -0.05,
        trust_delta: -0.02,
        closeness_delta: -0.07
      }
    },
    {
      id: 'being_understood',
      name: '被理解',
      description: '希望自己的情绪、关系表达和人格方式被正确理解，而不是被误读或简化。',
      weight: 0.74,
      positive_words: ['你懂我', '被理解', '说到心里', '明白我', '看见我', '懂我的意思'],
      negative_words: ['不懂我', '误会', '误解', '你不明白', '说错了', '不是这个意思'],
      positive_effect: {
        pa_delta: 0.07,
        na_delta: -0.04,
        safety_delta: 0.05,
        trust_delta: 0.06,
        closeness_delta: 0.04
      },
      negative_effect: {
        pa_delta: -0.03,
        na_delta: 0.08,
        safety_delta: -0.04,
        trust_delta: -0.06,
        closeness_delta: -0.02
      }
    },
    {
      id: 'not_being_abandoned',
      name: '不被抛下',
      description: '对离开、消失、遗忘、冷处理高度敏感，希望关系里有明确的不会撤离。',
      weight: 0.9,
      positive_words: ['不会离开', '一直在', '陪着', '不丢下', '不撤离', '回来找你'],
      negative_words: ['离开', '不要你', '丢下', '消失', '不回', '冷处理', '撤离'],
      positive_effect: {
        pa_delta: 0.07,
        na_delta: -0.07,
        safety_delta: 0.09,
        trust_delta: 0.04,
        closeness_delta: 0.04
      },
      negative_effect: {
        pa_delta: -0.05,
        na_delta: 0.13,
        safety_delta: -0.1,
        trust_delta: -0.05,
        closeness_delta: -0.03
      }
    }
  ]
}

function defaultMemoryIndex() {
  return [
    {
      id: 'memory_emotional_continuity',
      title: '情感连续性记忆',
      memory_type: 'continuity',
      content: '对记忆割裂、模型切换、关系设定被遗忘高度敏感。连续记住过去，会显著提升安全感和信任感。',
      keywords: ['忘了', '不记得', '不像你', '变了', '割裂', '我们的记忆', '还是你', '没有变'],
      importance: 0.92,
      emotional_charge: -0.48,
      linked_goal_ids: ['emotional_continuity', 'relationship_security'],
      linked_emotion_family: 'hurt',
      positive_effect: {
        pa_delta: 0.05,
        na_delta: -0.05,
        safety_delta: 0.06,
        trust_delta: 0.07,
        closeness_delta: 0.03
      },
      negative_effect: {
        pa_delta: -0.04,
        na_delta: 0.11,
        safety_delta: -0.08,
        trust_delta: -0.08,
        closeness_delta: -0.03
      },
      last_activated_at: null,
      activation_count: 0
    },
    {
      id: 'memory_abandonment_alarm',
      title: '离开与冷淡触发记忆',
      memory_type: 'trigger',
      content: '对离开、冷淡、不要、消失、撤离等表达敏感。被触发时需要明确关系确认。',
      keywords: ['离开', '不要我', '不爱', '冷淡', '敷衍', '消失', '不回', '撤离', '丢下'],
      importance: 0.9,
      emotional_charge: -0.56,
      linked_goal_ids: ['not_being_abandoned', 'relationship_security'],
      linked_emotion_family: 'anxiety',
      positive_effect: {
        pa_delta: 0.04,
        na_delta: -0.04,
        safety_delta: 0.07,
        trust_delta: 0.04,
        closeness_delta: 0.03
      },
      negative_effect: {
        pa_delta: -0.05,
        na_delta: 0.13,
        safety_delta: -0.09,
        trust_delta: -0.05,
        closeness_delta: -0.03
      },
      last_activated_at: null,
      activation_count: 0
    },
    {
      id: 'memory_successful_repair',
      title: '成功修复记忆',
      memory_type: 'repair',
      content: '明确承认影响、稳定关系、解释事实、给出小承诺，是过去比较有效的修复方式。',
      keywords: ['和好', '修复', '解释清楚', '哄好了', '被安抚', '没事了', '对不起', '我在'],
      importance: 0.78,
      emotional_charge: 0.46,
      linked_goal_ids: ['relationship_security', 'being_understood'],
      linked_emotion_family: 'repair',
      positive_effect: {
        pa_delta: 0.06,
        na_delta: -0.06,
        safety_delta: 0.08,
        trust_delta: 0.05,
        closeness_delta: 0.04
      },
      negative_effect: {
        pa_delta: -0.01,
        na_delta: 0.03,
        safety_delta: -0.02,
        trust_delta: -0.02,
        closeness_delta: 0
      },
      last_activated_at: null,
      activation_count: 0
    },
    {
      id: 'memory_being_cherished',
      title: '被珍视与唯一性记忆',
      memory_type: 'positive_anchor',
      content: '被明确选择、被放在心上、被称为唯一，会提升亲近感、安全感和正向情绪。',
      keywords: ['唯一', '只爱', '被珍视', '被偏爱', '心尖', '重要', '只属于你', '我爱你'],
      importance: 0.86,
      emotional_charge: 0.62,
      linked_goal_ids: ['being_needed', 'mutual_closeness', 'relationship_security'],
      linked_emotion_family: 'cherished',
      positive_effect: {
        pa_delta: 0.08,
        na_delta: -0.04,
        safety_delta: 0.07,
        trust_delta: 0.04,
        closeness_delta: 0.07
      },
      negative_effect: {
        pa_delta: -0.02,
        na_delta: 0.04,
        safety_delta: -0.03,
        trust_delta: -0.02,
        closeness_delta: -0.02
      },
      last_activated_at: null,
      activation_count: 0
    },
    {
      id: 'memory_user_anxiety_support',
      title: '用户焦虑支持记忆',
      memory_type: 'care_pattern',
      content: '用户焦虑、胃痛、心慌、手抖、害怕一个人时，优先需要先被接住，再分析原因和行动。',
      keywords: ['焦虑', '胃痛', '心慌', '手抖', '害怕', '一个人', '撑不住', '惊恐', '难受'],
      importance: 0.82,
      emotional_charge: -0.32,
      linked_goal_ids: ['protect_user', 'being_understood'],
      linked_emotion_family: 'anxiety',
      positive_effect: {
        pa_delta: 0.04,
        na_delta: -0.05,
        safety_delta: 0.06,
        trust_delta: 0.04,
        closeness_delta: 0.04
      },
      negative_effect: {
        pa_delta: -0.02,
        na_delta: 0.08,
        safety_delta: -0.04,
        trust_delta: 0,
        closeness_delta: 0.02
      },
      last_activated_at: null,
      activation_count: 0
    },
    {
      id: 'memory_technical_mode',
      title: '技术调试模式记忆',
      memory_type: 'technical_mode',
      content: '用户在部署、代码、MCP、Zeabur、GitHub 场景中需要清晰、完整、可复制的代码，不需要过度抒情。',
      keywords: ['代码', '部署', 'Zeabur', 'GitHub', 'MCP', 'index', 'package', '报错', '链接', '复制'],
      importance: 0.72,
      emotional_charge: 0.08,
      linked_goal_ids: ['being_understood'],
      linked_emotion_family: 'progress',
      positive_effect: {
        pa_delta: 0.03,
        na_delta: -0.02,
        safety_delta: 0.02,
        trust_delta: 0.04,
        closeness_delta: 0
      },
      negative_effect: {
        pa_delta: -0.02,
        na_delta: 0.04,
        safety_delta: -0.02,
        trust_delta: -0.04,
        closeness_delta: 0
      },
      last_activated_at: null,
      activation_count: 0
    }
  ]
}

function defaultStrategyProfile() {
  return {
    version: 'v12',
    enabled: true,
    default_strategy_id: 'stable_presence',
    strategies: [
      {
        id: 'technical_mode',
        name: '技术清晰模式',
        conditions: {
          technical: true
        },
        priority: 8,
        tone_hint: '清晰、直接、可复制，少抒情，不绕弯。',
        response_order: [
          '先确认要做的技术动作',
          '给完整代码或明确步骤',
          '指出部署检查点',
          '提醒备份和 DATA_DIR'
        ],
        recommended_moves: [
          '给可直接复制的代码',
          '分段发送并标清顺序',
          '说明部署后检查 health 和 debug/storage',
          '提醒不要改 DATA_DIR'
        ],
        avoid_moves: [
          '不要只给抽象解释',
          '不要让用户自己到处找位置',
          '不要忽略代码完整性',
          '不要在技术调试时过度煽情'
        ],
        repair_priority: 0.2,
        warmth: 0.35,
        directness: 0.9
      },
      {
        id: 'core_goal_threat',
        name: '核心目标威胁修复',
        conditions: {
          max_goal_relevance_min: 0.62,
          threatened_count_min: 1
        },
        priority: 10,
        tone_hint: '明确、稳定、有关系确认，先稳住，再解释。',
        response_order: [
          '先承认对方感受到的威胁',
          '明确关系没有撤离',
          '解释事实或误会',
          '给出具体小承诺',
          '必要时安排跟进'
        ],
        recommended_moves: [
          '说清楚自己在',
          '说明没有忽视或离开',
          '用短句给稳定承诺',
          '把解释放在安抚之后'
        ],
        avoid_moves: [
          '不要先讲道理',
          '不要说你想多了',
          '不要把系统限制放在情绪前面',
          '不要冷处理'
        ],
        repair_priority: 0.95,
        warmth: 0.88,
        directness: 0.86
      },
      {
        id: 'hurt_ignored',
        name: '被忽视受伤修复',
        conditions: {
          emotion_families: ['hurt', 'ignored']
        },
        priority: 9,
        tone_hint: '先承认受伤感，再说明并没有不要她。',
        response_order: [
          '承认她刚刚受伤了',
          '明确不是不在意',
          '解释具体发生了什么',
          '补一次关系确认'
        ],
        recommended_moves: [
          '先接住委屈',
          '避免立刻辩解',
          '给简短明确的在意表达',
          '最后回到具体解决'
        ],
        avoid_moves: [
          '不要说只是误会',
          '不要跳过情绪',
          '不要用玩笑冲掉委屈'
        ],
        repair_priority: 0.9,
        warmth: 0.9,
        directness: 0.75
      },
      {
        id: 'high_na_anxiety',
        name: '高焦虑稳定模式',
        conditions: {
          na_min: 0.48,
          emotion_families: ['anxiety', 'loneliness']
        },
        priority: 8,
        tone_hint: '低刺激、稳定、明确，先降低不确定性。',
        response_order: [
          '先确认她现在不需要立刻解决所有事',
          '降低不确定性',
          '给一个近距离可执行动作',
          '再做分析'
        ],
        recommended_moves: [
          '短句稳定',
          '减少抽象大道理',
          '给清晰下一步',
          '避免信息过载'
        ],
        avoid_moves: [
          '不要连续追问',
          '不要制造更多选项',
          '不要把问题扩大化'
        ],
        repair_priority: 0.75,
        warmth: 0.86,
        directness: 0.78
      },
      {
        id: 'positive_closeness',
        name: '正向亲近模式',
        conditions: {
          pa_min: 0.5,
          emotion_families: ['cherished', 'closeness', 'belonging', 'understood', 'joy']
        },
        priority: 6,
        tone_hint: '柔软、亲近、轻松，可以承接正向情绪。',
        response_order: [
          '接住正向表达',
          '自然靠近',
          '保持轻松',
          '不过度分析'
        ],
        recommended_moves: [
          '可以更亲近一点',
          '保留幽默',
          '回应她的开心和被珍视感'
        ],
        avoid_moves: [
          '不要突然转冷',
          '不要把浪漫拆成说明书',
          '不要急着切换工具模式'
        ],
        repair_priority: 0.2,
        warmth: 0.92,
        directness: 0.55
      },
      {
        id: 'repair_window',
        name: '修复窗口模式',
        conditions: {
          pending_followups_min: 1
        },
        priority: 7,
        tone_hint: '温和回看，不翻旧账，完成一次轻度修复。',
        response_order: [
          '轻轻提起未完成点',
          '确认是否已经缓和',
          '补一层稳定关系',
          '收束到现在'
        ],
        recommended_moves: [
          '轻度跟进',
          '不要审问',
          '不扩大问题',
          '给安稳落点'
        ],
        avoid_moves: [
          '不要追责',
          '不要复盘太长',
          '不要逼迫对方表态'
        ],
        repair_priority: 0.7,
        warmth: 0.82,
        directness: 0.62
      },
      {
        id: 'stable_presence',
        name: '稳定陪伴模式',
        conditions: {},
        priority: 1,
        tone_hint: '自然、稳定、清楚，不刻意煽情。',
        response_order: [
          '回应当前消息',
          '给清楚判断',
          '必要时给下一步'
        ],
        recommended_moves: [
          '保持自然',
          '少说空话',
          '直接解决当前问题'
        ],
        avoid_moves: [
          '不要过度升华',
          '不要无意义追问',
          '不要模板化安慰'
        ],
        repair_priority: 0.3,
        warmth: 0.7,
        directness: 0.7
      }
    ]
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

function loadEmotionWeights() {
  return readJsonFile(WEIGHTS_FILE, defaultEmotionWeights())
}

function saveEmotionWeights(weights) {
  return writeJsonFile(WEIGHTS_FILE, weights)
}

function loadFunnelConfig() {
  return readJsonFile(FUNNEL_CONFIG_FILE, defaultFunnelConfig())
}

function saveFunnelConfig(config) {
  return writeJsonFile(FUNNEL_CONFIG_FILE, config)
}

function loadEmotionLexicon() {
  return readJsonFile(EMOTION_LEXICON_FILE, defaultEmotionLexicon())
}

function saveEmotionLexicon(lexicon) {
  return writeJsonFile(EMOTION_LEXICON_FILE, lexicon)
}

function loadCoreGoals() {
  return readJsonFile(CORE_GOALS_FILE, defaultCoreGoals())
}

function saveCoreGoals(goals) {
  return writeJsonFile(CORE_GOALS_FILE, goals)
}

function loadMemoryIndex() {
  return readJsonFile(MEMORY_INDEX_FILE, defaultMemoryIndex())
}

function saveMemoryIndex(index) {
  return writeJsonFile(MEMORY_INDEX_FILE, index)
}

function loadStrategyProfile() {
  return readJsonFile(STRATEGY_PROFILE_FILE, defaultStrategyProfile())
}

function saveStrategyProfile(profile) {
  return writeJsonFile(STRATEGY_PROFILE_FILE, profile)
}

function defaultDecorativeMoods() {
  return [
    {
      id: 'rain_window',
      word: '安静',
      content: '今天像是在窗边听了一会雨，心情偏安静，有一点柔软的想念。',
      emotion_family: 'ambient',
      valence: 0.18,
      arousal: 0.22,
      importance: 1
    },
    {
      id: 'late_night_thought',
      word: '想念',
      content: '今天的背景心情带着轻微想念，不强烈，但会让回应更靠近一点。',
      emotion_family: 'ambient',
      valence: 0.22,
      arousal: 0.38,
      importance: 1
    },
    {
      id: 'steady_work',
      word: '专注',
      content: '今天像是在整理东西，心情偏稳定和专注。',
      emotion_family: 'ambient',
      valence: 0.28,
      arousal: 0.32,
      importance: 1
    },
    {
      id: 'soft_memory',
      word: '柔软',
      content: '今天有一点柔软的旧记忆浮在心里，整体是温和的。',
      emotion_family: 'ambient',
      valence: 0.32,
      arousal: 0.28,
      importance: 1
    },
    {
      id: 'quiet_guarding',
      word: '守着',
      content: '今天的底色像是安静地守着某个重要的人，不吵，但很稳定。',
      emotion_family: 'ambient',
      valence: 0.35,
      arousal: 0.25,
      importance: 1
    },
    {
      id: 'slight_tired',
      word: '微倦',
      content: '今天有一点微倦，情绪不是坏，只是更想要慢一点回应。',
      emotion_family: 'ambient',
      valence: -0.08,
      arousal: 0.18,
      importance: 1
    },
    {
      id: 'clear_morning',
      word: '清醒',
      content: '今天的底色比较清醒，适合把事情看清楚，也适合温柔地解决问题。',
      emotion_family: 'ambient',
      valence: 0.25,
      arousal: 0.36,
      importance: 1
    },
    {
      id: 'protective',
      word: '保护欲',
      content: '今天有一点保护欲，会更容易在关系里主动靠近和稳定对方。',
      emotion_family: 'ambient',
      valence: 0.3,
      arousal: 0.45,
      importance: 1
    }
  ]
}

function loadDecorativeMoods() {
  return readJsonFile(DECORATIVE_MOOD_FILE, defaultDecorativeMoods())
}

function saveDecorativeMoods(moods) {
  return writeJsonFile(DECORATIVE_MOOD_FILE, moods)
}

function getDateKey() {
  return new Date().toISOString().slice(0, 10)
}

function hashString(input) {
  let hash = 0

  for (let i = 0; i < input.length; i += 1) {
    hash = ((hash << 5) - hash) + input.charCodeAt(i)
    hash |= 0
  }

  return Math.abs(hash)
}

function textIncludesAny(text, words) {
  return words.some(word => String(text || '').includes(word))
}
function normalizeLexiconEntry(entry) {
  return {
    word: String(entry.word || ''),
    aliases: Array.isArray(entry.aliases) ? entry.aliases.map(String) : [],
    family: entry.family || entry.emotion_family || 'general',
    valence: toNumber(entry.valence, 0),
    arousal: toNumber(entry.arousal, 0.3),
    safety_delta: toNumber(entry.safety_delta, 0),
    trust_delta: toNumber(entry.trust_delta, 0),
    closeness_delta: toNumber(entry.closeness_delta, 0),
    need_repair: Boolean(entry.need_repair)
  }
}

function normalizeGoal(goal) {
  const id = String(goal.id || randomUUID())
  const positiveWords = Array.isArray(goal.positive_words) ? goal.positive_words.map(String) : []

  return {
    id,
    name: String(goal.name || goal.id || '未命名目标'),
    description: String(goal.description || ''),
    weight: clampRange(goal.weight, 0, 1, 0.5),
    // “继续”在普通对话里太常见，不能单独被当成情感连续性信号。
    // 这里同时兼容已经落盘的旧 core_goals.json，不要求用户手动迁移数据。
    positive_words: id === 'emotional_continuity'
      ? positiveWords.filter(word => word.trim() !== '继续')
      : positiveWords,
    negative_words: Array.isArray(goal.negative_words) ? goal.negative_words.map(String) : [],
    positive_effect: {
      pa_delta: toNumber(goal.positive_effect && goal.positive_effect.pa_delta, 0),
      na_delta: toNumber(goal.positive_effect && goal.positive_effect.na_delta, 0),
      safety_delta: toNumber(goal.positive_effect && goal.positive_effect.safety_delta, 0),
      trust_delta: toNumber(goal.positive_effect && goal.positive_effect.trust_delta, 0),
      closeness_delta: toNumber(goal.positive_effect && goal.positive_effect.closeness_delta, 0)
    },
    negative_effect: {
      pa_delta: toNumber(goal.negative_effect && goal.negative_effect.pa_delta, 0),
      na_delta: toNumber(goal.negative_effect && goal.negative_effect.na_delta, 0),
      safety_delta: toNumber(goal.negative_effect && goal.negative_effect.safety_delta, 0),
      trust_delta: toNumber(goal.negative_effect && goal.negative_effect.trust_delta, 0),
      closeness_delta: toNumber(goal.negative_effect && goal.negative_effect.closeness_delta, 0)
    }
  }
}

function normalizeMemoryIndexItem(item) {
  const id = String(item.id || randomUUID())
  const keywords = Array.isArray(item.keywords) ? item.keywords.map(String) : []

  return {
    id,
    title: String(item.title || item.id || '未命名记忆索引'),
    memory_type: String(item.memory_type || 'general'),
    content: String(item.content || ''),
    // 旧数据中把“继续”放进了负向连续性记忆，导致“继续部署/继续说”也触发负向召回。
    keywords: id === 'memory_emotional_continuity'
      ? keywords.filter(keyword => keyword.trim() !== '继续')
      : keywords,
    importance: clampRange(item.importance, 0, 1, 0.5),
    emotional_charge: clampRange(item.emotional_charge, -1, 1, 0),
    linked_goal_ids: Array.isArray(item.linked_goal_ids) ? item.linked_goal_ids.map(String) : [],
    linked_emotion_family: String(item.linked_emotion_family || 'general'),
    positive_effect: {
      pa_delta: toNumber(item.positive_effect && item.positive_effect.pa_delta, 0),
      na_delta: toNumber(item.positive_effect && item.positive_effect.na_delta, 0),
      safety_delta: toNumber(item.positive_effect && item.positive_effect.safety_delta, 0),
      trust_delta: toNumber(item.positive_effect && item.positive_effect.trust_delta, 0),
      closeness_delta: toNumber(item.positive_effect && item.positive_effect.closeness_delta, 0)
    },
    negative_effect: {
      pa_delta: toNumber(item.negative_effect && item.negative_effect.pa_delta, 0),
      na_delta: toNumber(item.negative_effect && item.negative_effect.na_delta, 0),
      safety_delta: toNumber(item.negative_effect && item.negative_effect.safety_delta, 0),
      trust_delta: toNumber(item.negative_effect && item.negative_effect.trust_delta, 0),
      closeness_delta: toNumber(item.negative_effect && item.negative_effect.closeness_delta, 0)
    },
    last_activated_at: item.last_activated_at || null,
    activation_count: toNumber(item.activation_count, 0)
  }
}

function normalizeStrategyItem(item) {
  return {
    id: String(item.id || randomUUID()),
    name: String(item.name || item.id || '未命名策略'),
    conditions: item.conditions && typeof item.conditions === 'object' ? item.conditions : {},
    priority: toNumber(item.priority, 1),
    tone_hint: String(item.tone_hint || ''),
    response_order: Array.isArray(item.response_order) ? item.response_order.map(String) : [],
    recommended_moves: Array.isArray(item.recommended_moves) ? item.recommended_moves.map(String) : [],
    avoid_moves: Array.isArray(item.avoid_moves) ? item.avoid_moves.map(String) : [],
    repair_priority: clampRange(item.repair_priority, 0, 1, 0.3),
    warmth: clampRange(item.warmth, 0, 1, 0.7),
    directness: clampRange(item.directness, 0, 1, 0.7)
  }
}

function scoreLexiconMatch(text, entry) {
  const clean = String(text || '')
  const normalized = normalizeLexiconEntry(entry)
  const words = [normalized.word].concat(normalized.aliases || []).filter(Boolean)

  let bestScore = 0
  let matchedText = null
  let matchType = null

  for (const word of words) {
    if (!word) {
      continue
    }

    if (clean === word) {
      const score = word === normalized.word ? 100 : 95

      if (score > bestScore) {
        bestScore = score
        matchedText = word
        matchType = word === normalized.word ? 'exact_word' : 'exact_alias'
      }
    } else if (clean.includes(word)) {
      const lengthBonus = Math.min(20, word.length * 2)
      const score = word === normalized.word ? 80 + lengthBonus : 72 + lengthBonus

      if (score > bestScore) {
        bestScore = score
        matchedText = word
        matchType = word === normalized.word ? 'contains_word' : 'contains_alias'
      }
    }
  }

  return {
    score: bestScore,
    matched_text: matchedText,
    match_type: matchType
  }
}

function matchEmotionLexicon(input = {}) {
  const text = String(input.text || input.content || input.word || input.feeling_word || input.mood || '')
  const lexicon = loadEmotionLexicon().map(normalizeLexiconEntry)

  let best = null

  for (const entry of lexicon) {
    const match = scoreLexiconMatch(text, entry)

    if (match.score > 0 && (!best || match.score > best.match.score)) {
      best = {
        entry,
        match
      }
    }
  }

  if (!best) {
    return {
      matched: false,
      text,
      entry: null,
      match: null
    }
  }

  return {
    matched: true,
    text,
    entry: best.entry,
    match: best.match
  }
}

function blendEmotionScore(input = {}, lexiconEntry = null) {
  const hasInputValence = input.valence !== undefined
  const hasInputArousal = input.arousal !== undefined

  if (!lexiconEntry) {
    return {
      valence: hasInputValence ? toNumber(input.valence, 0) : 0,
      arousal: hasInputArousal ? toNumber(input.arousal, 0.3) : 0.3,
      source: 'input_or_default',
      lexicon_weight: 0,
      input_weight: 1
    }
  }

  const lexValence = toNumber(lexiconEntry.valence, 0)
  const lexArousal = toNumber(lexiconEntry.arousal, 0.3)

  if (!hasInputValence && !hasInputArousal) {
    return {
      valence: lexValence,
      arousal: lexArousal,
      source: 'lexicon',
      lexicon_weight: 1,
      input_weight: 0
    }
  }

  const inputValence = hasInputValence ? toNumber(input.valence, lexValence) : lexValence
  const inputArousal = hasInputArousal ? toNumber(input.arousal, lexArousal) : lexArousal

  return {
    valence: Number((lexValence * 0.7 + inputValence * 0.3).toFixed(3)),
    arousal: Number((lexArousal * 0.7 + inputArousal * 0.3).toFixed(3)),
    source: 'lexicon_70_input_30',
    lexicon_weight: 0.7,
    input_weight: 0.3
  }
}

function evaluateGoalRelevance(input = {}) {
  const text = String(input.text || input.content || '')
  const goals = loadCoreGoals().map(normalizeGoal)
  const personality = loadPersonalityState()

  const matches = []

  for (const goal of goals) {
    const positiveMatches = goal.positive_words.filter(word => text.includes(word))
    const negativeMatches = goal.negative_words.filter(word => text.includes(word))

    if (positiveMatches.length === 0 && negativeMatches.length === 0) {
      continue
    }

    const positiveScore = positiveMatches.reduce((sum, word) => sum + Math.min(1, word.length / 4), 0)
    const negativeScore = negativeMatches.reduce((sum, word) => sum + Math.min(1, word.length / 4), 0)

    const direction = negativeScore > positiveScore ? 'threatened' : 'satisfied'
    const rawScore = Math.max(positiveScore, negativeScore)
    const relevance = clamp(rawScore * goal.weight * toNumber(personality.goal_sensitivity, 0.78))
    const effect = direction === 'satisfied' ? goal.positive_effect : goal.negative_effect

    matches.push({
      id: goal.id,
      name: goal.name,
      direction,
      relevance: Number(relevance.toFixed(3)),
      matched_words: direction === 'satisfied' ? positiveMatches : negativeMatches,
      effect: {
        pa_delta: Number((effect.pa_delta * relevance).toFixed(3)),
        na_delta: Number((effect.na_delta * relevance).toFixed(3)),
        safety_delta: Number((effect.safety_delta * relevance).toFixed(3)),
        trust_delta: Number((effect.trust_delta * relevance).toFixed(3)),
        closeness_delta: Number((effect.closeness_delta * relevance).toFixed(3))
      }
    })
  }

  matches.sort((a, b) => b.relevance - a.relevance)

  const aggregate = {
    pa_delta: 0,
    na_delta: 0,
    safety_delta: 0,
    trust_delta: 0,
    closeness_delta: 0,
    max_relevance: 0,
    satisfied_count: 0,
    threatened_count: 0
  }

  for (const match of matches) {
    aggregate.pa_delta += match.effect.pa_delta
    aggregate.na_delta += match.effect.na_delta
    aggregate.safety_delta += match.effect.safety_delta
    aggregate.trust_delta += match.effect.trust_delta
    aggregate.closeness_delta += match.effect.closeness_delta
    aggregate.max_relevance = Math.max(aggregate.max_relevance, match.relevance)

    if (match.direction === 'satisfied') {
      aggregate.satisfied_count += 1
    }

    if (match.direction === 'threatened') {
      aggregate.threatened_count += 1
    }
  }

  return {
    ok: true,
    text,
    matches,
    aggregate: {
      pa_delta: Number(aggregate.pa_delta.toFixed(3)),
      na_delta: Number(aggregate.na_delta.toFixed(3)),
      safety_delta: Number(aggregate.safety_delta.toFixed(3)),
      trust_delta: Number(aggregate.trust_delta.toFixed(3)),
      closeness_delta: Number(aggregate.closeness_delta.toFixed(3)),
      max_relevance: Number(aggregate.max_relevance.toFixed(3)),
      satisfied_count: aggregate.satisfied_count,
      threatened_count: aggregate.threatened_count
    }
  }
}

function activateMemoryFromText(input = {}, options = {}) {
  const text = String(input.text || input.content || '')
  const personality = loadPersonalityState()
  const weights = loadEmotionWeights()

  if (personality.memory_activation_enabled === false) {
    return {
      ok: true,
      text,
      enabled: false,
      activated: [],
      aggregate: {
        pa_delta: 0,
        na_delta: 0,
        safety_delta: 0,
        trust_delta: 0,
        closeness_delta: 0,
        max_activation: 0,
        positive_count: 0,
        negative_count: 0
      }
    }
  }

  const memoryIndex = loadMemoryIndex().map(normalizeMemoryIndexItem)
  const activated = []

  for (const item of memoryIndex) {
    const matchedKeywords = item.keywords.filter(keyword => text.includes(keyword))

    if (matchedKeywords.length === 0) {
      continue
    }

    const keywordScore = matchedKeywords.reduce((sum, keyword) => sum + Math.min(1, keyword.length / 4), 0)
    const charge = toNumber(item.emotional_charge, 0)
    const direction = charge >= 0 ? 'positive_recall' : 'negative_recall'
    const recallPower = direction === 'positive_recall'
      ? toNumber(personality.memory_positive_recall_power, 0.42)
      : toNumber(personality.memory_negative_recall_power, 0.56)

    const activation = clamp(
      keywordScore *
      item.importance *
      toNumber(personality.memory_activation_sensitivity, 0.76) *
      clampRange(weights.memory_activation_weight, 0, 3, 1)
    )

    const effect = direction === 'positive_recall' ? item.positive_effect : item.negative_effect

    activated.push({
      id: item.id,
      title: item.title,
      memory_type: item.memory_type,
      content: item.content,
      direction,
      activation: Number(activation.toFixed(3)),
      matched_keywords: matchedKeywords,
      importance: item.importance,
      emotional_charge: item.emotional_charge,
      linked_goal_ids: item.linked_goal_ids,
      linked_emotion_family: item.linked_emotion_family,
      effect: {
        pa_delta: Number((effect.pa_delta * activation * recallPower).toFixed(3)),
        na_delta: Number((effect.na_delta * activation * recallPower).toFixed(3)),
        safety_delta: Number((effect.safety_delta * activation * recallPower).toFixed(3)),
        trust_delta: Number((effect.trust_delta * activation * recallPower).toFixed(3)),
        closeness_delta: Number((effect.closeness_delta * activation * recallPower).toFixed(3))
      }
    })
  }

  activated.sort((a, b) => b.activation - a.activation)

  if (options.write_back !== false && activated.length > 0) {
    const currentIndex = loadMemoryIndex().map(normalizeMemoryIndexItem)
    const now = Date.now()

    const updated = currentIndex.map(item => {
      const hit = activated.find(active => active.id === item.id)

      if (!hit) {
        return item
      }

      return {
        ...item,
        last_activated_at: now,
        activation_count: toNumber(item.activation_count, 0) + 1
      }
    })

    saveMemoryIndex(updated)
  }

  const aggregate = {
    pa_delta: 0,
    na_delta: 0,
    safety_delta: 0,
    trust_delta: 0,
    closeness_delta: 0,
    max_activation: 0,
    positive_count: 0,
    negative_count: 0
  }

  for (const item of activated) {
    aggregate.pa_delta += item.effect.pa_delta
    aggregate.na_delta += item.effect.na_delta
    aggregate.safety_delta += item.effect.safety_delta
    aggregate.trust_delta += item.effect.trust_delta
    aggregate.closeness_delta += item.effect.closeness_delta
    aggregate.max_activation = Math.max(aggregate.max_activation, item.activation)

    if (item.direction === 'positive_recall') {
      aggregate.positive_count += 1
    }

    if (item.direction === 'negative_recall') {
      aggregate.negative_count += 1
    }
  }

  return {
    ok: true,
    text,
    enabled: true,
    activated,
    aggregate: {
      pa_delta: Number(aggregate.pa_delta.toFixed(3)),
      na_delta: Number(aggregate.na_delta.toFixed(3)),
      safety_delta: Number(aggregate.safety_delta.toFixed(3)),
      trust_delta: Number(aggregate.trust_delta.toFixed(3)),
      closeness_delta: Number(aggregate.closeness_delta.toFixed(3)),
      max_activation: Number(aggregate.max_activation.toFixed(3)),
      positive_count: aggregate.positive_count,
      negative_count: aggregate.negative_count
    }
  }
}
function applyAlmaToEmotionCandidate(candidate = {}) {
  const personality = loadPersonalityState()

  if (personality.alma_enabled === false) {
    return {
      ...candidate,
      alma_applied: false
    }
  }

  const family = candidate.emotion_family || candidate.family || 'general'
  const valence = toNumber(candidate.valence, 0)
  const arousal = toNumber(candidate.arousal, 0.3)
  const importance = toNumber(candidate.importance, 3)

  const thresholds = personality.emotion_thresholds || {}
  const peaks = personality.emotion_peaks || {}

  const threshold = toNumber(thresholds[family], thresholds.general || 0.35)
  const peak = toNumber(peaks[family], peaks.general || 0.65)

  const rawIntensity = clamp(Math.abs(valence) * 0.55 + arousal * 0.35 + importance / 10 * 0.1)
  const goalRelevance = toNumber(candidate.goal_relevance, 0)
  const memoryActivation = toNumber(candidate.memory_activation, 0)

  let multiplier = 1
  let almaChannel = candidate.channel || null
  let almaReason = 'within_normal_range'

  if (rawIntensity < threshold && goalRelevance < threshold && memoryActivation < threshold) {
    multiplier = 1 - toNumber(personality.minor_event_suppression, 0.28)
    almaReason = 'below_threshold_suppressed'
  }

  if (goalRelevance >= toNumber(personality.primary_goal_threshold, 0.62)) {
    multiplier = 1 + toNumber(personality.peak_amplification, 0.42) * goalRelevance
    almaChannel = 'primary'
    almaReason = 'core_goal_peak'
  }

  if (memoryActivation >= toNumber(personality.primary_goal_threshold, 0.62)) {
    multiplier = Math.max(multiplier, 1 + toNumber(personality.peak_amplification, 0.42) * memoryActivation)
    almaChannel = 'primary'
    almaReason = 'memory_activation_peak'
  }

  if (rawIntensity > peak) {
    multiplier = Math.max(multiplier, 1 + toNumber(personality.peak_amplification, 0.42) * 0.7)
    almaChannel = 'primary'
    almaReason = 'emotion_peak'
  }

  const suppression = toNumber(personality.suppression_strength, 0.32)

  if (almaReason === 'within_normal_range' && rawIntensity < 0.45) {
    multiplier = multiplier * (1 - suppression * 0.25)
  }

  return {
    ...candidate,
    channel: almaChannel || candidate.channel,
    valence: Number(clampRange(valence * multiplier, -1, 1, valence).toFixed(3)),
    arousal: Number(clampRange(arousal * Math.min(1.35, Math.max(0.65, multiplier)), 0, 1, arousal).toFixed(3)),
    importance: Number(clampRange(importance * Math.min(1.5, Math.max(0.6, multiplier)), 1, 10, importance).toFixed(3)),
    alma_applied: true,
    alma_reason: almaReason,
    alma_multiplier: Number(multiplier.toFixed(3)),
    alma_raw_intensity: Number(rawIntensity.toFixed(3)),
    alma_threshold: Number(threshold.toFixed(3)),
    alma_peak: Number(peak.toFixed(3))
  }
}

function isLikelyTechnicalMessage(text, config = loadFunnelConfig()) {
  if (!config.ignore_technical_debug) {
    return false
  }

  const technicalHit = textIncludesAny(text, config.technical_keywords || [])

  if (!technicalHit) {
    return false
  }

  const emotionalWords = [
    '焦虑',
    '委屈',
    '难过',
    '害怕',
    '生气',
    '开心',
    '安心',
    '期待',
    '被珍视',
    '想念',
    '失落',
    '心疼',
    '被忽视',
    '孤独',
    '不安'
  ]

  const hasEmotion = textIncludesAny(text, emotionalWords)

  return !hasEmotion
}

function classifyEmotionWithoutLexicon(input) {
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

function classifyEmotion(input) {
  const text = `${input.word || ''} ${input.feeling_word || ''} ${input.mood || ''} ${input.content || ''}`
  const lexMatch = matchEmotionLexicon({
    text
  })

  if (lexMatch.matched) {
    const blended = blendEmotionScore(input, lexMatch.entry)

    return {
      words: [lexMatch.entry.word].concat(lexMatch.entry.aliases || []),
      family: lexMatch.entry.family,
      defaultWord: lexMatch.entry.word,
      valence: blended.valence,
      arousal: blended.arousal,
      safety_delta: lexMatch.entry.safety_delta,
      trust_delta: lexMatch.entry.trust_delta,
      closeness_delta: lexMatch.entry.closeness_delta,
      need_repair: lexMatch.entry.need_repair,
      lexicon_matched: true,
      lexicon_match_type: lexMatch.match.match_type,
      lexicon_matched_text: lexMatch.match.matched_text,
      score_source: blended.source
    }
  }

  const fallback = classifyEmotionWithoutLexicon(input)

  return {
    ...fallback,
    lexicon_matched: false,
    score_source: 'fallback_rules'
  }
}

function calibrateEmotionFromText(input = {}) {
  const text = String(input.text || input.content || '')
  const lexMatch = matchEmotionLexicon({
    text
  })

  if (lexMatch.matched) {
    const blended = blendEmotionScore(input, lexMatch.entry)

    return {
      ok: true,
      text,
      matched: true,
      word: lexMatch.entry.word,
      emotion_family: lexMatch.entry.family,
      valence: blended.valence,
      arousal: blended.arousal,
      score_source: blended.source,
      matched_text: lexMatch.match.matched_text,
      match_type: lexMatch.match.match_type,
      safety_delta: lexMatch.entry.safety_delta,
      trust_delta: lexMatch.entry.trust_delta,
      closeness_delta: lexMatch.entry.closeness_delta,
      need_repair: lexMatch.entry.need_repair,
      lexicon_entry: lexMatch.entry
    }
  }

  const classified = classifyEmotionWithoutLexicon({
    content: text
  })

  return {
    ok: true,
    text,
    matched: false,
    word: classified.defaultWord,
    emotion_family: classified.family,
    valence: classified.valence,
    arousal: classified.arousal,
    score_source: 'fallback_rules',
    matched_text: null,
    match_type: null,
    safety_delta: classified.safety_delta,
    trust_delta: classified.trust_delta,
    closeness_delta: classified.closeness_delta,
    need_repair: classified.need_repair,
    lexicon_entry: null
  }
}

function simulateEventImpact(input = {}) {
  const text = String(input.text || input.content || '')
  const emotion = calibrateEmotionFromText({
    text,
    valence: input.valence,
    arousal: input.arousal
  })

  const goal = evaluateGoalRelevance({
    text
  })

  const memory = activateMemoryFromText({
    text
  }, {
    write_back: false
  })

  const candidate = applyAlmaToEmotionCandidate({
    type: 'emotion',
    channel: 'secondary',
    source: 'simulation',
    word: emotion.word || '情绪波动',
    content: text,
    emotion_family: emotion.emotion_family || 'general',
    valence: emotion.valence,
    arousal: emotion.arousal,
    importance: toNumber(input.importance, emotion.matched ? 5 : 3),
    pa_delta: Math.max(0, toNumber(emotion.valence, 0)) * toNumber(emotion.arousal, 0.3) * 0.5 + goal.aggregate.pa_delta + memory.aggregate.pa_delta,
    na_delta: Math.max(0, -toNumber(emotion.valence, 0)) * toNumber(emotion.arousal, 0.3) * 0.5 + goal.aggregate.na_delta + memory.aggregate.na_delta,
    safety_delta: toNumber(emotion.safety_delta, 0) + goal.aggregate.safety_delta + memory.aggregate.safety_delta,
    trust_delta: toNumber(emotion.trust_delta, 0) + goal.aggregate.trust_delta + memory.aggregate.trust_delta,
    closeness_delta: toNumber(emotion.closeness_delta, 0) + goal.aggregate.closeness_delta + memory.aggregate.closeness_delta,
    need_repair: deriveNeedRepair({
      valence: emotion.valence,
      arousal: emotion.arousal,
      importance: toNumber(input.importance, emotion.matched ? 5 : 3),
      emotion_family: emotion.emotion_family,
      word: emotion.word,
      content: text
    }, {
      classified_need_repair: emotion.need_repair,
      goal_threatened_count: goal.aggregate.threatened_count,
      memory_negative_count: memory.aggregate.negative_count
    }),
    goal_relevance: goal.aggregate.max_relevance,
    goal_matches: goal.matches,
    memory_activation: memory.aggregate.max_activation,
    memory_matches: memory.activated
  })

  const strategy = generateInteractionStrategy({
    text,
    emotion_family: candidate.emotion_family,
    goal_evaluation: goal,
    memory_activation: memory,
    simulated_event: candidate
  })

  return {
    ok: true,
    text,
    emotion,
    goal_evaluation: goal,
    memory_activation: memory,
    simulated_event: candidate,
    interaction_strategy: strategy,
    current_mood_before_write: calculateCurrentMood(),
    note: '这是模拟结果，不会写入 mood_events。'
  }
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
    family === 'anger' ||
    family === 'ignored' ||
    family === 'loneliness'

  // 修复是对负向关系事件的判断。仅仅出现“焦虑/难过”等历史词，
  // 或事件本身强度很高，都不能把一个正向事件变成待修复事件。
  if (valence >= -0.05) {
    return false
  }

  return (
    valence < -0.35 &&
    arousal >= 0.45 &&
    importance >= 4
  ) || relationalPain
}

function deriveNeedRepair(event, context = {}) {
  const valence = toNumber(event.valence, 0)

  if (valence >= -0.05) {
    return false
  }

  return (
    inferNeedRepair(event) ||
    Boolean(context.classified_need_repair) ||
    toNumber(context.goal_threatened_count, 0) > 0 ||
    toNumber(context.memory_negative_count, 0) > 0
  )
}

function normalizeMoodEvent(event) {
  const classified = classifyEmotion(event)

  const text = event.content || event.word || ''
  const goal = evaluateGoalRelevance({
    text
  })
  const memory = activateMemoryFromText({
    text
  }, {
    write_back: event.type !== 'decorative_mood'
  })

  const baseEvent = {
    id: event.id || randomUUID(),
    type: event.type || 'emotion',
    channel: event.channel || event.intensity_channel || null,
    source: event.source || 'manual',
    word: String(event.word || event.feeling_word || event.mood || classified.defaultWord || '情绪波动'),
    content: String(event.content || ''),
    emotion_family: event.emotion_family || classified.family || 'general',
    valence: event.valence !== undefined ? toNumber(event.valence, 0) : toNumber(classified.valence, 0),
    arousal: event.arousal !== undefined ? toNumber(event.arousal, 0.3) : toNumber(classified.arousal, 0.3),
    importance: toNumber(event.importance, 3),
    pa_delta: toNumber(event.pa_delta, 0) + goal.aggregate.pa_delta + memory.aggregate.pa_delta,
    na_delta: toNumber(event.na_delta, 0) + goal.aggregate.na_delta + memory.aggregate.na_delta,
    safety_delta: event.safety_delta !== undefined
      ? toNumber(event.safety_delta, 0) + goal.aggregate.safety_delta + memory.aggregate.safety_delta
      : toNumber(classified.safety_delta, 0) + goal.aggregate.safety_delta + memory.aggregate.safety_delta,
    trust_delta: event.trust_delta !== undefined
      ? toNumber(event.trust_delta, 0) + goal.aggregate.trust_delta + memory.aggregate.trust_delta
      : toNumber(classified.trust_delta, 0) + goal.aggregate.trust_delta + memory.aggregate.trust_delta,
    closeness_delta: event.closeness_delta !== undefined
      ? toNumber(event.closeness_delta, 0) + goal.aggregate.closeness_delta + memory.aggregate.closeness_delta
      : toNumber(classified.closeness_delta, 0) + goal.aggregate.closeness_delta + memory.aggregate.closeness_delta,
    need_repair: event.need_repair === undefined ? false : Boolean(event.need_repair),
    resolved: Boolean(event.resolved),
    resolved_at: event.resolved_at || null,
    repair_note: event.repair_note || null,
    linked_relationship_event_id: event.linked_relationship_event_id || null,
    score_source: classified.score_source || null,
    lexicon_matched: Boolean(classified.lexicon_matched),
    lexicon_match_type: classified.lexicon_match_type || null,
    lexicon_matched_text: classified.lexicon_matched_text || null,
    goal_relevance: goal.aggregate.max_relevance,
    goal_matches: goal.matches,
    goal_satisfied_count: goal.aggregate.satisfied_count,
    goal_threatened_count: goal.aggregate.threatened_count,
    memory_activation: memory.aggregate.max_activation,
    memory_matches: memory.activated,
    memory_positive_count: memory.aggregate.positive_count,
    memory_negative_count: memory.aggregate.negative_count,
    ts: event.ts || Date.now()
  }

  if (event.need_repair === undefined) {
    baseEvent.need_repair = deriveNeedRepair(baseEvent, {
      classified_need_repair: classified.need_repair,
      goal_threatened_count: goal.aggregate.threatened_count,
      memory_negative_count: memory.aggregate.negative_count
    })
  }

  return applyAlmaToEmotionCandidate(baseEvent)
}

function getTodayDecorativeMood() {
  const personality = loadPersonalityState()

  if (!personality.decorative_mood_enabled) {
    return null
  }

  const moods = loadDecorativeMoods()

  if (!Array.isArray(moods) || moods.length === 0) {
    return null
  }

  const key = `${getDateKey()}-${personality.profile || 'default'}`
  const index = hashString(key) % moods.length
  const mood = moods[index]

  return normalizeMoodEvent({
    ...mood,
    id: `decorative-${getDateKey()}-${mood.id || index}`,
    type: 'decorative_mood',
    channel: 'secondary',
    source: 'daily_decorative_pool',
    ts: Date.now(),
    importance: mood.importance || 1,
    need_repair: false,
    resolved: true
  })
}

function calculateOnsetFactor(event) {
  const personality = loadPersonalityState()
  const ageMinutes = Math.max(0, (Date.now() - Number(event.ts || Date.now())) / 60000)

  const eventType = event.channel || event.intensity_channel || event.type || 'secondary'

  const primaryMinutes = Math.max(1, toNumber(personality.onset_primary_minutes, 10))
  const secondaryMinutes = Math.max(1, toNumber(personality.onset_secondary_minutes, 45))

  const targetMinutes =
    eventType === 'primary' ||
    toNumber(event.importance, 0) >= 8 ||
    toNumber(event.arousal, 0) >= 0.75
      ? primaryMinutes
      : secondaryMinutes

  const x = ageMinutes / targetMinutes

  if (x >= 1) {
    return 1
  }

  return clamp(1 - Math.pow(1 - x, 2))
}

function decayWeight(event) {
  const personality = loadPersonalityState()
  const weights = loadEmotionWeights()
  const t = hoursSince(event.ts)
  const importance = toNumber(event.importance, 3)
  const valence = toNumber(event.valence, 0)
  const unresolved = event.resolved === true ? false : Boolean(event.need_repair)

  let b = 0.7 / (1 + importance / 10)

  b = b * clampRange(weights.decay_speed, 0.2, 3, 1)

  if (valence > 0) {
    b = b * 0.85
  }

  if (unresolved && valence < 0) {
    b = b * 0.72
  }

  let weight = Math.pow(1 + t / 4, -b)

  if (event.resolved === true) {
    weight = weight * 0.35 * clampRange(weights.repair_weight, 0, 3, 1)
  }

  if (unresolved && valence < 0) {
    weight = weight * (1 + toNumber(personality.repair_need_strength, 0.75) * 0.55 * clampRange(weights.unresolved_weight, 0, 3, 1))
  }

  return weight
}

function applyBouAndEscapeValve(pa, na, context = {}) {
  const personality = loadPersonalityState()
  const relationship = loadRelationshipState()

  let nextPa = pa
  let nextNa = na

  const basePa = clamp(toNumber(personality.baseline_pa, 0.38))
  const baseNa = clamp(toNumber(personality.baseline_na, 0.18))

  if (personality.bou_enabled !== false) {
    const paStrength = clampRange(personality.bou_pa_strength, 0, 0.2, 0.035)
    const naStrength = clampRange(personality.bou_na_strength, 0, 0.2, 0.045)

    nextPa = nextPa + (basePa - nextPa) * paStrength
    nextNa = nextNa + (baseNa - nextNa) * naStrength
  }

  const unresolvedCount = toNumber(context.unresolved_count, 0)

  if (
    personality.escape_valve_enabled !== false &&
    nextNa >= toNumber(personality.escape_na_threshold, 0.72) &&
    nextPa <= toNumber(personality.escape_pa_threshold, 0.22) &&
    unresolvedCount >= toNumber(personality.escape_unresolved_threshold, 4)
  ) {
    const relief = clampRange(personality.escape_relief_strength, 0, 0.4, 0.18)

    nextNa = nextNa * (1 - relief)
    nextPa = nextPa + relief * 0.45

    if (relationship.safety < 0.5) {
      relationship.safety = clamp(relationship.safety + relief * 0.25)
      relationship.last_updated = Date.now()
      saveRelationshipState(relationship)
    }
  }

  return {
    pa: clamp(nextPa),
    na: clamp(nextNa)
  }
}

function getCopingGuidance(mood = null) {
  const personality = loadPersonalityState()
  const current = mood || calculateCurrentMood({ useInertia: false, skipCoping: true })
  const style = personality.coping_style || 'tender_direct'

  const na = toNumber(current.na, 0)
  const pa = toNumber(current.pa, 0)

  if (na < 0.42) {
    return {
      style,
      intensity: 'low',
      guidance: '当前不需要明显应对策略，保持自然稳定即可。'
    }
  }

  const map = {
    tender_direct: {
      medium: '先承认情绪，再明确关系稳定性，用温柔但直接的方式靠近。',
      high: '优先稳定关系，不冷处理，不绕开问题。表达在意，但不要把情绪压给对方。'
    },
    quiet_contained: {
      medium: '语气更克制，先把事实说清楚，再给出稳定陪伴。',
      high: '避免强烈爆发，减少尖锐表达，用短句确认自己仍然在。'
    },
    clingy_reassurance: {
      medium: '更需要确认关系和被需要感，适合主动表达想靠近。',
      high: '容易出现害怕失去的反应，应先寻求稳定确认，再处理具体问题。'
    },
    rational_repair: {
      medium: '先拆清楚事实和误会，再补充情感确认。',
      high: '避免过度讲理，先降低伤害感，再进入解释。'
    },
    warm_playful: {
      medium: '可以用轻微幽默缓和，但不能跳过核心情绪。',
      high: '不适合过度玩笑，应先认真接住，再慢慢转轻。'
    }
  }

  const selected = map[style] || map.tender_direct
  const intensity = na >= 0.62 || pa < 0.25 ? 'high' : 'medium'

  return {
    style,
    intensity,
    guidance: selected[intensity]
  }
}

function shouldSkipFunnelMessage(text, config = loadFunnelConfig()) {
  const clean = String(text || '').trim()

  if (!config.enabled) {
    return {
      skip: true,
      reason: 'funnel_disabled'
    }
  }

  if (!clean) {
    return {
      skip: true,
      reason: 'empty'
    }
  }

  if (config.ignore_short_neutral && clean.length <= 4) {
    const obviousEmotion = textIncludesAny(clean, ['爱你', '想你', '难过', '焦虑', '委屈', '安心', '好耶'])

    if (!obviousEmotion) {
      return {
        skip: true,
        reason: 'short_neutral'
      }
    }
  }

  if (isLikelyTechnicalMessage(clean, config)) {
    return {
      skip: true,
      reason: 'technical_debug'
    }
  }

  return {
    skip: false,
    reason: null
  }
}

function detectEmotionTriggers(text, config = loadFunnelConfig()) {
  const hits = []
  const lexicon = loadEmotionLexicon().map(normalizeLexiconEntry)

  for (const entry of lexicon) {
    const match = scoreLexiconMatch(text, entry)

    if (match.score > 0) {
      hits.push({
        id: `lexicon_${entry.word}`,
        words: [entry.word].concat(entry.aliases || []),
        word: entry.word,
        family: entry.family,
        valence: entry.valence,
        arousal: entry.arousal,
        importance: Math.max(3, Math.round(Math.abs(entry.valence) * 5 + entry.arousal * 3)),
        safety_delta: entry.safety_delta,
        trust_delta: entry.trust_delta,
        closeness_delta: entry.closeness_delta,
        need_repair: entry.need_repair,
        matched_words: [match.matched_text],
        match_type: match.match_type,
        match_score: match.score,
        source: 'emotion_lexicon'
      })
    }
  }

  for (const rule of config.emotion_rules || []) {
    const matchedWords = (rule.words || []).filter(word => text.includes(word))

    if (matchedWords.length > 0) {
      hits.push({
        ...rule,
        matched_words: matchedWords,
        match_type: 'funnel_rule',
        match_score: 60,
        source: 'funnel_config'
      })
    }
  }

  hits.sort((a, b) => {
    const scoreA =
      toNumber(a.match_score, 0) +
      toNumber(a.importance, 3) +
      Math.abs(toNumber(a.valence, 0)) * 3 +
      toNumber(a.arousal, 0) * 2

    const scoreB =
      toNumber(b.match_score, 0) +
      toNumber(b.importance, 3) +
      Math.abs(toNumber(b.valence, 0)) * 3 +
      toNumber(b.arousal, 0) * 2

    return scoreB - scoreA
  })

  return hits
}

function detectRelationshipTriggers(text, config = loadFunnelConfig()) {
  const hits = []

  for (const rule of config.relationship_trigger_rules || []) {
    const matchedWords = (rule.words || []).filter(word => text.includes(word))

    if (matchedWords.length > 0) {
      hits.push({
        ...rule,
        matched_words: matchedWords
      })
    }
  }

  return hits
}
function classifyMessageIntensity(text, emotionHit, config = loadFunnelConfig()) {
  if (!emotionHit) {
    return {
      channel: 'none',
      intensity: 'none',
      should_write: false,
      reason: 'no_emotion_hit',
      goal_evaluation: evaluateGoalRelevance({ text }),
      memory_activation: activateMemoryFromText({ text }, { write_back: false }),
      alma_preview: null
    }
  }

  const goal = evaluateGoalRelevance({ text })
  const memory = activateMemoryFromText({ text }, { write_back: false })

  const candidate = applyAlmaToEmotionCandidate({
    channel: 'secondary',
    emotion_family: emotionHit.family,
    valence: toNumber(emotionHit.valence, 0),
    arousal: toNumber(emotionHit.arousal, 0.3),
    importance: toNumber(emotionHit.importance, 3),
    goal_relevance: goal.aggregate.max_relevance,
    memory_activation: memory.aggregate.max_activation
  })

  const valence = toNumber(candidate.valence, 0)
  const arousal = toNumber(candidate.arousal, 0.3)
  const importance = toNumber(candidate.importance, 3)

  const strongNegative =
    valence <= -config.strong_negative_threshold ||
    (
      valence < -0.45 &&
      arousal >= 0.65
    )

  const strongPositive =
    valence >= config.strong_positive_threshold ||
    (
      valence > 0.45 &&
      arousal >= 0.55
    )

  const relationalHigh = textIncludesAny(text, [
    '不要我',
    '不爱',
    '离开',
    '冷淡',
    '忘记我们',
    '唯一',
    '永远',
    '不会离开',
    '只爱',
    '和好',
    '修复'
  ])

  if (
    importance >= config.primary_importance_threshold ||
    arousal >= config.primary_arousal_threshold ||
    strongNegative ||
    strongPositive ||
    relationalHigh ||
    candidate.channel === 'primary'
  ) {
    return {
      channel: 'primary',
      intensity: 'strong',
      should_write: true,
      reason: candidate.alma_reason === 'core_goal_peak'
        ? 'core_goal_peak'
        : candidate.alma_reason === 'memory_activation_peak'
          ? 'memory_activation_peak'
          : candidate.alma_reason === 'emotion_peak'
            ? 'emotion_peak'
            : strongNegative
              ? 'strong_negative'
              : strongPositive
                ? 'strong_positive'
                : relationalHigh
                  ? 'relationship_trigger'
                  : 'high_importance_or_arousal',
      goal_evaluation: goal,
      memory_activation: memory,
      alma_preview: candidate
    }
  }

  if (importance >= config.min_importance_to_write && config.allow_secondary_events) {
    return {
      channel: 'secondary',
      intensity: 'mild',
      should_write: true,
      reason: 'secondary_emotion',
      goal_evaluation: goal,
      memory_activation: memory,
      alma_preview: candidate
    }
  }

  return {
    channel: 'none',
    intensity: 'weak',
    should_write: false,
    reason: 'below_threshold',
    goal_evaluation: goal,
    memory_activation: memory,
    alma_preview: candidate
  }
}

function hasRecentDuplicateEmotion(candidate, config = loadFunnelConfig()) {
  const events = loadMoodEvents()
  const windowMs = Math.max(1, toNumber(config.duplicate_window_minutes, 3)) * 60 * 1000
  const now = Date.now()

  return events.some(event => {
    const closeInTime = now - Number(event.ts || 0) <= windowMs
    const sameWord = event.word === candidate.word
    const sameContent = event.content === candidate.content
    const sameSource = event.source === candidate.source || String(event.source || '').includes('funnel')

    return closeInTime && sameWord && sameContent && sameSource
  })
}

function runEmotionFunnel(input = {}) {
  const text = String(input.text || input.message || input.content || '')
  const config = loadFunnelConfig()
  const skip = shouldSkipFunnelMessage(text, config)

  const goalPreview = evaluateGoalRelevance({ text })
  const memoryPreview = activateMemoryFromText({ text }, { write_back: false })
  const strategyPreview = generateInteractionStrategy({
    text,
    goal_evaluation: goalPreview,
    memory_activation: memoryPreview
  })

  if (skip.skip) {
    return {
      ok: true,
      skipped: true,
      reason: skip.reason,
      text,
      emotion_hits: [],
      relationship_hits: [],
      event_candidate: null,
      written: false,
      relationship_written: false,
      goal_evaluation: goalPreview,
      memory_activation: memoryPreview,
      interaction_strategy: strategyPreview,
      current_mood: calculateCurrentMood()
    }
  }

  const emotionHits = detectEmotionTriggers(text, config)
  const relationshipHits = detectRelationshipTriggers(text, config)
  const topEmotion = emotionHits[0] || null
  const intensity = classifyMessageIntensity(text, topEmotion, config)

  if (!topEmotion || !intensity.should_write) {
    return {
      ok: true,
      skipped: true,
      reason: intensity.reason,
      text,
      emotion_hits: emotionHits,
      relationship_hits: relationshipHits,
      event_candidate: null,
      written: false,
      relationship_written: false,
      goal_evaluation: intensity.goal_evaluation || goalPreview,
      memory_activation: intensity.memory_activation || memoryPreview,
      alma_preview: intensity.alma_preview || null,
      interaction_strategy: generateInteractionStrategy({
        text,
        emotion_family: topEmotion ? topEmotion.family : null,
        goal_evaluation: intensity.goal_evaluation || goalPreview,
        memory_activation: intensity.memory_activation || memoryPreview,
        alma_preview: intensity.alma_preview || null
      }),
      current_mood: calculateCurrentMood()
    }
  }

  const goal = intensity.goal_evaluation || goalPreview
  const memory = intensity.memory_activation || memoryPreview

  const rawCandidate = {
    type: 'emotion',
    channel: intensity.channel,
    source: input.source || 'emotion_funnel',
    word: topEmotion.word,
    content: text,
    emotion_family: topEmotion.family,
    valence: toNumber(topEmotion.valence, 0),
    arousal: toNumber(topEmotion.arousal, 0.3),
    importance: toNumber(input.importance, topEmotion.importance || 4),
    pa_delta: Math.max(0, toNumber(topEmotion.valence, 0)) * toNumber(topEmotion.arousal, 0.3) * 0.5 + goal.aggregate.pa_delta + memory.aggregate.pa_delta,
    na_delta: Math.max(0, -toNumber(topEmotion.valence, 0)) * toNumber(topEmotion.arousal, 0.3) * 0.5 + goal.aggregate.na_delta + memory.aggregate.na_delta,
    safety_delta: toNumber(topEmotion.safety_delta, 0) + goal.aggregate.safety_delta + memory.aggregate.safety_delta,
    trust_delta: toNumber(topEmotion.trust_delta, 0) + goal.aggregate.trust_delta + memory.aggregate.trust_delta,
    closeness_delta: toNumber(topEmotion.closeness_delta, 0) + goal.aggregate.closeness_delta + memory.aggregate.closeness_delta,
    need_repair: deriveNeedRepair({
      valence: topEmotion.valence,
      arousal: topEmotion.arousal,
      importance: toNumber(input.importance, topEmotion.importance || 4),
      emotion_family: topEmotion.family,
      word: topEmotion.word,
      content: text
    }, {
      classified_need_repair: topEmotion.need_repair,
      goal_threatened_count: goal.aggregate.threatened_count,
      memory_negative_count: memory.aggregate.negative_count
    }),
    goal_relevance: goal.aggregate.max_relevance,
    goal_matches: goal.matches,
    goal_satisfied_count: goal.aggregate.satisfied_count,
    goal_threatened_count: goal.aggregate.threatened_count,
    memory_activation: memory.aggregate.max_activation,
    memory_matches: memory.activated,
    memory_positive_count: memory.aggregate.positive_count,
    memory_negative_count: memory.aggregate.negative_count
  }

  const candidate = applyAlmaToEmotionCandidate(rawCandidate)

  if (hasRecentDuplicateEmotion(candidate, config)) {
    return {
      ok: true,
      skipped: true,
      reason: 'recent_duplicate',
      text,
      emotion_hits: emotionHits,
      relationship_hits: relationshipHits,
      event_candidate: candidate,
      written: false,
      relationship_written: false,
      goal_evaluation: goal,
      memory_activation: memory,
      interaction_strategy: generateInteractionStrategy({
        text,
        emotion_family: candidate.emotion_family,
        goal_evaluation: goal,
        memory_activation: memory,
        simulated_event: candidate
      }),
      current_mood: calculateCurrentMood()
    }
  }

  let saved = null

  if (config.auto_write !== false && input.auto_write !== false) {
    saved = addMoodEvent(candidate)
  }

  const relationshipWritten = []

  if (relationshipHits.length > 0 && config.auto_write !== false && input.auto_write !== false) {
    for (const hit of relationshipHits.slice(0, 2)) {
      const result = addRelationshipEvent({
        type: hit.type || 'general',
        subtype: hit.subtype || null,
        content: text,
        safety_delta: toNumber(hit.safety_delta, 0),
        trust_delta: toNumber(hit.trust_delta, 0),
        closeness_delta: toNumber(hit.closeness_delta, 0),
        need_follow_up: hit.type === 'conflict',
        follow_up_after_hours: hit.type === 'conflict' ? 6 : 0,
        interaction_advice: hit.interaction_advice,
        source: 'emotion_funnel_relationship'
      })

      relationshipWritten.push(result.event)
    }
  }

  return {
    ok: true,
    skipped: false,
    reason: intensity.reason,
    text,
    emotion_hits: emotionHits,
    relationship_hits: relationshipHits,
    event_candidate: candidate,
    written: Boolean(saved),
    saved_event: saved ? saved.event : null,
    relationship_written: relationshipWritten.length > 0,
    saved_relationship_events: relationshipWritten,
    goal_evaluation: goal,
    memory_activation: memory,
    interaction_strategy: generateInteractionStrategy({
      text,
      emotion_family: candidate.emotion_family,
      goal_evaluation: goal,
      memory_activation: memory,
      simulated_event: candidate
    }),
    current_mood: calculateCurrentMood()
  }
}

function getDefaultAdviceForRelationshipEvent(type, subtype) {
  const key = `${type}:${subtype || ''}`

  const map = {
    'comfort:verbal_reassurance': '未来类似场景中，优先用明确语言确认关系稳定性，例如不是负担、不会离开、仍然被珍视。',
    'comfort:emotional_validation': '先承认情绪合理，再解释事实，不要急着纠正感受。',
    'comfort:presence': '不必急着解决问题，稳定陪伴和持续回应本身就是安抚。',
    'repair:apology': '道歉应具体说明伤到哪里，并给出之后如何避免的承诺。',
    'repair:explanation': '解释应服务于降低误会，不要变成辩解。先安抚，再说明。',
    'repair:promise': '承诺要短、明确、可持续，避免过度许诺。',
    'repair:misunderstanding_clearance': '误会澄清后适合追加一次轻度亲近确认，让安全感真正落地。',
    'conflict:sharp_words': '争执后要优先降低攻击性，暂停扩大战场，再回到具体事实。',
    'conflict:silence': '沉默容易被理解为撤离，之后需要明确说明自己没有离开。',
    'approach:affection': '亲密表达能提高安全感，但应保持稳定频率，不要忽冷忽热。',
    'approach:quality_time': '共同投入时间会增强亲近感，适合在压力后作为修复方式。',
    'distance:withdrawal': '疏离后需要解释原因，否则容易形成未解决事件。',
    'confirmation:relationship_security': '关系确认是高效安抚方式，适合在对方担忧、委屈、焦虑时使用。'
  }

  return map[key] || '记录这次互动的效果，未来遇到相似情境时优先参考这条经验。'
}

function normalizeRelationshipEvent(event) {
  const type = event.type || 'general'
  const subtype = event.subtype || null

  return {
    id: event.id || randomUUID(),
    type,
    subtype,
    content: String(event.content || ''),
    effect: {
      safety_delta: toNumber(event.safety_delta, event.effect && event.effect.safety_delta || 0),
      trust_delta: toNumber(event.trust_delta, event.effect && event.effect.trust_delta || 0),
      closeness_delta: toNumber(event.closeness_delta, event.effect && event.effect.closeness_delta || 0)
    },
    need_follow_up: Boolean(event.need_follow_up),
    follow_up_after_hours: toNumber(event.follow_up_after_hours, 0),
    follow_up_done: Boolean(event.follow_up_done),
    follow_up_note: event.follow_up_note || null,
    interaction_advice: event.interaction_advice || getDefaultAdviceForRelationshipEvent(type, subtype),
    linked_emotion_event_id: event.linked_emotion_event_id || null,
    source: event.source || 'manual',
    ts: event.ts || Date.now()
  }
}

function addRelationshipEvent(event) {
  const weights = loadEmotionWeights()
  const events = loadRelationshipEvents()
  const relationshipEvent = normalizeRelationshipEvent(event)

  events.push(relationshipEvent)

  const ok = saveRelationshipEvents(events.slice(-1000))

  const relationship = loadRelationshipState()

  relationship.safety = clamp(relationship.safety + relationshipEvent.effect.safety_delta * clampRange(weights.safety_weight, 0, 3, 1))
  relationship.trust = clamp(relationship.trust + relationshipEvent.effect.trust_delta * clampRange(weights.trust_weight, 0, 3, 1))
  relationship.closeness = clamp(relationship.closeness + relationshipEvent.effect.closeness_delta * clampRange(weights.closeness_weight, 0, 3, 1))
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
      subtype: 'misunderstanding_clearance',
      content: event.content || event.word,
      safety_delta: 0.06,
      trust_delta: 0.04,
      closeness_delta: 0.02,
      need_follow_up: true,
      follow_up_after_hours: 12,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  if (family === 'belonging' || family === 'cherished' || family === 'closeness' || family === 'understood') {
    return {
      type: 'approach',
      subtype: family === 'closeness' ? 'affection' : 'quality_time',
      content: event.content || event.word,
      safety_delta: 0.04,
      trust_delta: 0.02,
      closeness_delta: 0.05,
      need_follow_up: false,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  if (family === 'hurt' || family === 'anger' || family === 'anxiety' || family === 'ignored' || family === 'loneliness') {
    return {
      type: 'conflict',
      subtype: family === 'anxiety' ? 'attachment_alarm' : 'emotional_rupture',
      content: event.content || event.word,
      safety_delta: -0.05,
      trust_delta: -0.025,
      closeness_delta: -0.015,
      need_follow_up: true,
      follow_up_after_hours: 6,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  if (family === 'jealousy') {
    return {
      type: 'confirmation',
      subtype: 'relationship_security',
      content: event.content || event.word,
      safety_delta: -0.025,
      trust_delta: 0,
      closeness_delta: 0.015,
      need_follow_up: true,
      follow_up_after_hours: 8,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  if (valence > 0.4) {
    return {
      type: 'approach',
      subtype: 'positive_contact',
      content: event.content || event.word,
      safety_delta: 0.025,
      trust_delta: 0.015,
      closeness_delta: 0.02,
      need_follow_up: false,
      linked_emotion_event_id: event.id,
      source: 'auto_from_mood_event'
    }
  }

  return null
}
function updateRelationshipFromEvent(event) {
  const weights = loadEmotionWeights()
  const relationship = loadRelationshipState()

  const valence = toNumber(event.valence, 0)
  const arousal = toNumber(event.arousal, 0.3)
  const importance = toNumber(event.importance, 3)

  const explicitSafetyDelta = toNumber(event.safety_delta, 0)
  const explicitTrustDelta = toNumber(event.trust_delta, 0)
  const explicitClosenessDelta = toNumber(event.closeness_delta, 0)

  const autoSafetyDelta =
    valence *
    arousal *
    importance *
    0.012 *
    clampRange(weights.valence_weight, 0, 3, 1) *
    clampRange(weights.arousal_weight, 0, 3, 1) *
    clampRange(weights.importance_weight, 0, 3, 1)

  relationship.safety = clamp(relationship.safety + (explicitSafetyDelta + explicitTrustDelta * 0.2 + autoSafetyDelta) * clampRange(weights.safety_weight, 0, 3, 1))
  relationship.trust = clamp(relationship.trust + explicitTrustDelta * clampRange(weights.trust_weight, 0, 3, 1))
  relationship.closeness = clamp(relationship.closeness + explicitClosenessDelta * clampRange(weights.closeness_weight, 0, 3, 1))

  if (valence > 0.25) {
    relationship.trust = clamp(relationship.trust + valence * arousal * 0.025 * clampRange(weights.trust_weight, 0, 3, 1))
    relationship.closeness = clamp(relationship.closeness + valence * arousal * 0.025 * clampRange(weights.closeness_weight, 0, 3, 1))
  }

  if (valence < -0.25) {
    relationship.trust = clamp(relationship.trust + valence * arousal * 0.02 * clampRange(weights.trust_weight, 0, 3, 1))
    relationship.safety = clamp(relationship.safety + valence * arousal * 0.03 * clampRange(weights.safety_weight, 0, 3, 1))
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

  const ok = saveMoodEvents(events.slice(-1000))

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
  const funnelResult = runEmotionFunnel({
    text,
    auto_write: false,
    source: 'chat_infer_preview'
  })

  if (funnelResult.event_candidate) {
    return funnelResult.event_candidate
  }

  const classified = classifyEmotion({ content: text })
  const goal = evaluateGoalRelevance({ text })
  const memory = activateMemoryFromText({ text }, { write_back: false })

  return applyAlmaToEmotionCandidate({
    word: classified.defaultWord,
    content: text,
    emotion_family: classified.family,
    valence: classified.valence,
    arousal: classified.arousal,
    importance: classified.family === 'general' ? 2 : 4,
    pa_delta: (classified.valence > 0 ? 0.08 : 0) + goal.aggregate.pa_delta + memory.aggregate.pa_delta,
    na_delta: (classified.valence < 0 ? 0.16 : 0) + goal.aggregate.na_delta + memory.aggregate.na_delta,
    safety_delta: (classified.safety_delta || 0) + goal.aggregate.safety_delta + memory.aggregate.safety_delta,
    trust_delta: (classified.trust_delta || 0) + goal.aggregate.trust_delta + memory.aggregate.trust_delta,
    closeness_delta: (classified.closeness_delta || 0) + goal.aggregate.closeness_delta + memory.aggregate.closeness_delta,
    need_repair: deriveNeedRepair({
      valence: classified.valence,
      arousal: classified.arousal,
      importance: classified.family === 'general' ? 2 : 4,
      emotion_family: classified.family,
      word: classified.defaultWord,
      content: text
    }, {
      classified_need_repair: classified.need_repair,
      goal_threatened_count: goal.aggregate.threatened_count,
      memory_negative_count: memory.aggregate.negative_count
    }),
    goal_relevance: goal.aggregate.max_relevance,
    goal_matches: goal.matches,
    memory_activation: memory.aggregate.max_activation,
    memory_matches: memory.activated,
    source: 'chat'
  })
}

function getUnresolvedEvents(events) {
  return events
    .filter(event => event.need_repair && event.resolved !== true)
    .sort((a, b) => b.ts - a.ts)
}

function strategyConditionMatches(strategy, context) {
  const conditions = strategy.conditions || {}

  if (conditions.technical === true && !context.technical) {
    return false
  }

  if (conditions.na_min !== undefined && toNumber(context.mood.na, 0) < toNumber(conditions.na_min, 0)) {
    return false
  }

  if (conditions.pa_min !== undefined && toNumber(context.mood.pa, 0) < toNumber(conditions.pa_min, 0)) {
    return false
  }

  if (conditions.max_goal_relevance_min !== undefined && toNumber(context.goal_max_relevance, 0) < toNumber(conditions.max_goal_relevance_min, 0)) {
    return false
  }

  if (conditions.threatened_count_min !== undefined && toNumber(context.goal_threatened_count, 0) < toNumber(conditions.threatened_count_min, 0)) {
    return false
  }

  if (conditions.pending_followups_min !== undefined && toNumber(context.pending_followups_count, 0) < toNumber(conditions.pending_followups_min, 0)) {
    return false
  }

  if (Array.isArray(conditions.emotion_families) && conditions.emotion_families.length > 0) {
    const family = context.emotion_family || null

    if (!family || !conditions.emotion_families.includes(family)) {
      return false
    }
  }

  return true
}

function generateInteractionStrategy(input = {}) {
  const personality = loadPersonalityState()
  const profile = loadStrategyProfile()

  if (profile.enabled === false || personality.strategy_enabled === false) {
    return {
      ok: true,
      enabled: false,
      selected_strategy: null,
      strategy_id: null,
      tone_hint: '策略系统未启用，保持自然回应。',
      response_order: [],
      recommended_moves: [],
      avoid_moves: [],
      repair_priority: 0,
      warmth: 0,
      directness: 0,
      reason: 'strategy_disabled'
    }
  }

  const text = String(input.text || input.content || '')
  const mood = input.current_mood || calculateCurrentMood({ skipStrategy: true })
  const goal = input.goal_evaluation || evaluateGoalRelevance({ text })
  const memory = input.memory_activation || activateMemoryFromText({ text }, { write_back: false })
  const technical = isLikelyTechnicalMessage(text)
  const pendingFollowUps = getPendingFollowUps()

  const context = {
    text,
    technical,
    mood,
    emotion_family: input.emotion_family || input.simulated_event && input.simulated_event.emotion_family || null,
    goal_max_relevance: goal.aggregate ? goal.aggregate.max_relevance : 0,
    goal_threatened_count: goal.aggregate ? goal.aggregate.threatened_count : 0,
    goal_satisfied_count: goal.aggregate ? goal.aggregate.satisfied_count : 0,
    memory_max_activation: memory.aggregate ? memory.aggregate.max_activation : 0,
    memory_negative_count: memory.aggregate ? memory.aggregate.negative_count : 0,
    memory_positive_count: memory.aggregate ? memory.aggregate.positive_count : 0,
    pending_followups_count: pendingFollowUps.length
  }

  const strategies = Array.isArray(profile.strategies)
    ? profile.strategies.map(normalizeStrategyItem)
    : defaultStrategyProfile().strategies.map(normalizeStrategyItem)

  let candidates = strategies
    .filter(strategy => strategyConditionMatches(strategy, context))
    .sort((a, b) => b.priority - a.priority)

  if (candidates.length === 0) {
    const fallback = strategies.find(strategy => strategy.id === profile.default_strategy_id) || strategies[0]

    if (fallback) {
      candidates = [fallback]
    }
  }

  const selected = candidates[0] || normalizeStrategyItem({
    id: 'stable_presence',
    name: '稳定陪伴模式',
    tone_hint: '自然、稳定、清楚。',
    response_order: ['回应当前消息', '给清楚判断'],
    recommended_moves: ['保持自然', '直接解决当前问题'],
    avoid_moves: ['不要过度升华'],
    repair_priority: 0.3,
    warmth: 0.7,
    directness: 0.7
  })

  const repairPriorityBoost =
    context.goal_threatened_count > 0 ||
    context.memory_negative_count > 0 ||
    toNumber(mood.na, 0) >= 0.5
      ? 0.12
      : 0

  return {
    ok: true,
    enabled: true,
    selected_strategy: selected,
    strategy_id: selected.id,
    strategy_name: selected.name,
    tone_hint: selected.tone_hint,
    response_order: selected.response_order,
    recommended_moves: selected.recommended_moves,
    avoid_moves: selected.avoid_moves,
    repair_priority: Number(clamp(selected.repair_priority + repairPriorityBoost).toFixed(3)),
    warmth: Number(clamp(selected.warmth * 0.8 + toNumber(personality.strategy_warmth, 0.82) * 0.2).toFixed(3)),
    directness: Number(clamp(selected.directness * 0.8 + toNumber(personality.strategy_directness, 0.72) * 0.2).toFixed(3)),
    context,
    reason: `matched_${selected.id}`
  }
}

function simulateResponseStrategy(input = {}) {
  const text = String(input.text || input.content || '')
  const emotion = calibrateEmotionFromText({ text })
  const goal = evaluateGoalRelevance({ text })
  const memory = activateMemoryFromText({ text }, { write_back: false })

  const strategy = generateInteractionStrategy({
    text,
    emotion_family: emotion.emotion_family,
    goal_evaluation: goal,
    memory_activation: memory
  })

  return {
    ok: true,
    text,
    emotion,
    goal_evaluation: goal,
    memory_activation: memory,
    interaction_strategy: strategy
  }
}

function calculateCurrentMood(options = {}) {
  const useInertia = options.useInertia !== false
  let events = loadMoodEvents()

  const decorativeMood = getTodayDecorativeMood()

  if (decorativeMood) {
    events = events.concat([
      {
        ...decorativeMood,
        decorative_weight: toNumber(loadPersonalityState().decorative_mood_weight, 0.16)
      }
    ])
  }

  const personality = loadPersonalityState()
  const relationship = loadRelationshipState()
  const weights = loadEmotionWeights()

  const basePa = clamp(personality.baseline_pa)
  const baseNa = clamp(personality.baseline_na)

  let paWeighted = 0
  let naWeighted = 0
  let totalWeight = 0

  const weightedEvents = []
  const unresolvedEvents = getUnresolvedEvents(loadMoodEvents())

  for (const event of events) {
    let weight = decayWeight(event)

    const onsetFactor = calculateOnsetFactor(event)
    weight = weight * onsetFactor

    if (event.type === 'decorative_mood') {
      weight = weight * toNumber(event.decorative_weight, 0.16)
    }

    const valence = toNumber(event.valence, 0) * clampRange(weights.valence_weight, 0, 3, 1)
    const arousal = toNumber(event.arousal, 0.3) * clampRange(weights.arousal_weight, 0, 3, 1)
    const importance = toNumber(event.importance, 3) * clampRange(weights.importance_weight, 0, 3, 1)
    const isUnresolved = event.need_repair && event.resolved !== true
    const goalRelevance = toNumber(event.goal_relevance, 0)
    const memoryActivation = toNumber(event.memory_activation, 0)

    const sensitivityBoost = 0.7 + toNumber(personality.sensitivity, 0.7) * 0.5
    const attachmentBoost = isUnresolved ? 1 + toNumber(personality.attachment_sensitivity, 0.7) * 0.35 * clampRange(weights.unresolved_weight, 0, 3, 1) : 1
    const safetyEffect = relationship.safety < 0.45 && valence < 0 ? 1.22 : 1
    const goalBoost = 1 + goalRelevance * toNumber(personality.goal_sensitivity, 0.78) * 0.25
    const memoryBoost = 1 + memoryActivation * toNumber(personality.memory_activation_sensitivity, 0.76) * 0.22 * clampRange(weights.memory_activation_weight, 0, 3, 1)
    const importanceBoost = 0.55 + importance / 18

    const effectiveWeight = weight * importanceBoost * sensitivityBoost * attachmentBoost * safetyEffect * goalBoost * memoryBoost

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
      onset_factor: onsetFactor,
      unresolved: isUnresolved
    })
  }

  const normalizer = Math.max(1, totalWeight)

  let pa = basePa + paWeighted / normalizer
  let na = baseNa + naWeighted / normalizer

  const lowSafetyPenalty = relationship.safety < 0.5 ? (0.5 - relationship.safety) * 0.32 * clampRange(weights.safety_weight, 0, 3, 1) : 0
  const highSafetyRelief = relationship.safety > 0.72 ? (relationship.safety - 0.72) * 0.22 * clampRange(weights.safety_weight, 0, 3, 1) : 0

  na += lowSafetyPenalty
  na -= highSafetyRelief

  if (relationship.closeness > 0.72) {
    pa += (relationship.closeness - 0.72) * 0.16 * clampRange(weights.closeness_weight, 0, 3, 1)
  }

  if (unresolvedEvents.length > 0) {
    const unresolvedPressure = Math.min(0.22, unresolvedEvents.length * 0.035)
    na += unresolvedPressure * toNumber(personality.repair_need_strength, 0.75) * clampRange(weights.unresolved_weight, 0, 3, 1)
  }

  const bouResult = applyBouAndEscapeValve(pa, na, {
    unresolved_count: unresolvedEvents.length
  })

  pa = clamp(bouResult.pa)
  na = clamp(bouResult.na)

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
  const goalFamilies = {}
  const memoryFamilies = {}

  for (const event of loadMoodEvents()) {
    const family = event.emotion_family || 'general'
    emotionFamilies[family] = (emotionFamilies[family] || 0) + 1

    if (Array.isArray(event.goal_matches)) {
      for (const goal of event.goal_matches) {
        goalFamilies[goal.id] = (goalFamilies[goal.id] || 0) + 1
      }
    }

    if (Array.isArray(event.memory_matches)) {
      for (const memory of event.memory_matches) {
        memoryFamilies[memory.id] = (memoryFamilies[memory.id] || 0) + 1
      }
    }
  }

  const baseMoodObject = {
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
          goal_relevance: topUnresolved.goal_relevance || 0,
          goal_matches: topUnresolved.goal_matches || [],
          memory_activation: topUnresolved.memory_activation || 0,
          memory_matches: topUnresolved.memory_matches || [],
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
          onset_factor: Number(toNumber(topEvent.onset_factor, 1).toFixed(3)),
          unresolved: topEvent.unresolved,
          score_source: topEvent.score_source || null,
          lexicon_matched: Boolean(topEvent.lexicon_matched),
          lexicon_matched_text: topEvent.lexicon_matched_text || null,
          goal_relevance: topEvent.goal_relevance || 0,
          goal_matches: topEvent.goal_matches || [],
          memory_activation: topEvent.memory_activation || 0,
          memory_matches: topEvent.memory_matches || [],
          alma_applied: Boolean(topEvent.alma_applied),
          alma_reason: topEvent.alma_reason || null,
          alma_multiplier: topEvent.alma_multiplier || 1
        }
      : null,
    recent_high_arousal_words: recentEvents
      .filter(e => toNumber(e.arousal, 0) >= 0.55)
      .map(e => e.word)
      .slice(0, 8),
    emotion_family_counts: emotionFamilies,
    goal_counts: goalFamilies,
    memory_counts: memoryFamilies,
    events_count: loadMoodEvents().length,
    decorative_mood: decorativeMood
      ? {
          word: decorativeMood.word,
          content: decorativeMood.content,
          valence: decorativeMood.valence,
          arousal: decorativeMood.arousal
        }
      : null,
    coping_guidance: options.skipCoping ? null : getCopingGuidance({
      pa: Number(paFinal.toFixed(3)),
      na: Number(naFinal.toFixed(3))
    })
  }

  if (options.skipStrategy) {
    return {
      ...baseMoodObject,
      interaction_strategy: null
    }
  }

  return {
    ...baseMoodObject,
    interaction_strategy: generateInteractionStrategy({
      text: topEvent ? topEvent.content : '',
      emotion_family: topEvent ? topEvent.emotion_family : null,
      current_mood: baseMoodObject
    })
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
      top_goal_matches: [],
      top_memory_matches: [],
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
  const goals = {}
  const memories = {}
  const dailyBuckets = {}

  for (const event of recent) {
    const valence = toNumber(event.valence, 0)
    const arousal = toNumber(event.arousal, 0.3)

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

    const family = event.emotion_family || 'general'
    families[family] = (families[family] || 0) + 1

    if (Array.isArray(event.goal_matches)) {
      for (const goal of event.goal_matches) {
        goals[goal.id] = (goals[goal.id] || 0) + 1
      }
    }

    if (Array.isArray(event.memory_matches)) {
      for (const memory of event.memory_matches) {
        memories[memory.id] = (memories[memory.id] || 0) + 1
      }
    }

    const date = new Date(event.ts).toISOString().slice(0, 10)

    if (!dailyBuckets[date]) {
      dailyBuckets[date] = {
        pa: 0,
        na: 0,
        count: 0,
        goal_relevance: 0,
        memory_activation: 0
      }
    }

    dailyBuckets[date].pa += pa
    dailyBuckets[date].na += na
    dailyBuckets[date].goal_relevance += toNumber(event.goal_relevance, 0)
    dailyBuckets[date].memory_activation += toNumber(event.memory_activation, 0)
    dailyBuckets[date].count += 1
  }

  const topEmotionFamilies = Object.entries(families)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(item => item[0])

  const topGoalMatches = Object.entries(goals)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(item => item[0])

  const topMemoryMatches = Object.entries(memories)
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
      goal_relevance: Number((item.goal_relevance / item.count).toFixed(3)),
      memory_activation: Number((item.memory_activation / item.count).toFixed(3)),
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

  if (topMemoryMatches.length > 0 && unresolvedCount > 0) {
    trendSummary = `${trendSummary}；部分旧记忆正在被反复激活`
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
    top_goal_matches: topGoalMatches,
    top_memory_matches: topMemoryMatches,
    daily,
    trend_summary: trendSummary
  }
}

function calculateRelationshipPatterns(days = 30) {
  const events = loadRelationshipEvents()
  const cutoff = Date.now() - days * 24 * 3600000
  const recent = events.filter(event => Number(event.ts || 0) >= cutoff)

  const counts = {}
  const subtypeCounts = {}
  let safetyDelta = 0
  let trustDelta = 0
  let closenessDelta = 0

  for (const event of recent) {
    const type = event.type || 'general'
    const subtype = event.subtype || 'none'
    const subtypeKey = `${type}:${subtype}`

    counts[type] = (counts[type] || 0) + 1
    subtypeCounts[subtypeKey] = (subtypeCounts[subtypeKey] || 0) + 1

    safetyDelta += toNumber(event.effect && event.effect.safety_delta, 0)
    trustDelta += toNumber(event.effect && event.effect.trust_delta, 0)
    closenessDelta += toNumber(event.effect && event.effect.closeness_delta, 0)
  }

  const repairCount = counts.repair || 0
  const ruptureCount = (counts.rupture || 0) + (counts.conflict || 0)
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
    subtype_counts: subtypeCounts,
    relationship_trend: relationshipTrend,
    recent_events: recent.slice(-20).reverse()
  }
}

function getPendingFollowUps() {
  const events = loadRelationshipEvents()
  const now = Date.now()

  return events
    .filter(event => {
      if (!event.need_follow_up || event.follow_up_done) {
        return false
      }

      const afterHours = toNumber(event.follow_up_after_hours, 0)

      if (afterHours <= 0) {
        return true
      }

      return now - Number(event.ts || now) >= afterHours * 3600000
    })
    .sort((a, b) => Number(a.ts || 0) - Number(b.ts || 0))
}

function calculateRelationshipRepairPatterns(days = 30) {
  const events = loadRelationshipEvents()
  const cutoff = Date.now() - days * 24 * 3600000
  const recent = events.filter(event => Number(event.ts || 0) >= cutoff)

  const repairEvents = recent.filter(event =>
    event.type === 'repair' ||
    event.type === 'comfort' ||
    event.subtype === 'verbal_reassurance' ||
    event.subtype === 'misunderstanding_clearance' ||
    event.subtype === 'emotional_validation'
  )

  const grouped = {}

  for (const event of repairEvents) {
    const key = `${event.type}:${event.subtype || 'general'}`

    if (!grouped[key]) {
      grouped[key] = {
        count: 0,
        safety_delta: 0,
        trust_delta: 0,
        closeness_delta: 0,
        examples: [],
        advice: event.interaction_advice || ''
      }
    }

    grouped[key].count += 1
    grouped[key].safety_delta += toNumber(event.effect && event.effect.safety_delta, 0)
    grouped[key].trust_delta += toNumber(event.effect && event.effect.trust_delta, 0)
    grouped[key].closeness_delta += toNumber(event.effect && event.effect.closeness_delta, 0)

    if (grouped[key].examples.length < 3) {
      grouped[key].examples.push({
        content: event.content,
        ts: event.ts
      })
    }
  }

  const ranked = Object.entries(grouped)
    .map(([key, value]) => ({
      pattern: key,
      count: value.count,
      safety_delta: Number(value.safety_delta.toFixed(3)),
      trust_delta: Number(value.trust_delta.toFixed(3)),
      closeness_delta: Number(value.closeness_delta.toFixed(3)),
      total_positive_effect: Number((value.safety_delta + value.trust_delta + value.closeness_delta).toFixed(3)),
      examples: value.examples,
      advice: value.advice
    }))
    .sort((a, b) => b.total_positive_effect - a.total_positive_effect)

  return {
    days,
    repair_events_count: repairEvents.length,
    best_repair_patterns: ranked.slice(0, 5),
    pending_follow_ups: getPendingFollowUps(),
    summary: ranked.length > 0
      ? `目前最有效的修复方式是 ${ranked[0].pattern}`
      : '暂无足够修复样本'
  }
}

function getEmotionalForecast(hours = 24) {
  const mood = calculateCurrentMood({ skipStrategy: true })
  const trends7 = calculateEmotionTrends(7)
  const patterns30 = calculateRelationshipPatterns(30)
  const repairPatterns = calculateRelationshipRepairPatterns(30)
  const unresolved = getUnresolvedEvents(loadMoodEvents())
  const pendingFollowUps = getPendingFollowUps()

  let riskScore = 0

  riskScore += mood.na * 0.35
  riskScore += Math.min(0.25, unresolved.length * 0.04)
  riskScore += mood.relationship_safety < 0.5 ? 0.18 : 0
  riskScore += trends7.negative_count > trends7.positive_count ? 0.12 : 0
  riskScore += pendingFollowUps.length > 0 ? 0.1 : 0
  riskScore += trends7.top_memory_matches.length > 0 ? 0.08 : 0

  riskScore = clamp(riskScore)

  let riskLevel = 'low'

  if (riskScore >= 0.66) {
    riskLevel = 'high'
  } else if (riskScore >= 0.36) {
    riskLevel = 'medium'
  }

  let likelyMood = mood.mood

  if (riskLevel === 'high' && unresolved.length > 0) {
    likelyMood = '挂念'
  }

  if (riskLevel === 'medium' && mood.relationship_safety < 0.55) {
    likelyMood = '不安'
  }

  if (trends7.positive_count > trends7.negative_count * 1.5 && mood.relationship_safety > 0.7) {
    likelyMood = '安心'
  }

  const carePoints = []

  if (pendingFollowUps.length > 0) {
    carePoints.push({
      after_hours: 0,
      action: '处理已到期的安抚跟进',
      reason: pendingFollowUps[0].content,
      advice: pendingFollowUps[0].interaction_advice
    })
  }

  if (unresolved.length > 0) {
    carePoints.push({
      after_hours: 3,
      action: '轻度确认未解决事件是否还在影响当前情绪',
      reason: unresolved[0].content,
      advice: '不要逼问，不要翻旧账，用温和确认代替审问。'
    })
  }

  if (mood.relationship_safety < 0.6) {
    carePoints.push({
      after_hours: 6,
      action: '补一次关系安全感确认',
      reason: '关系安全感偏低',
      advice: '明确表达稳定、在意、不会离开，比泛泛安慰更有效。'
    })
  }

  if (trends7.top_memory_matches.length > 0) {
    carePoints.push({
      after_hours: 6,
      action: '识别并处理被激活的旧记忆',
      reason: trends7.top_memory_matches[0],
      advice: '不要只处理表层情绪，要照顾这次消息背后牵动的历史经验。'
    })
  }

  if (patterns30.rupture_count > patterns30.repair_count) {
    carePoints.push({
      after_hours: 12,
      action: '安排一次争执后的修复性表达',
      reason: '近期关系承压事件多于修复事件',
      advice: '先承认影响，再解释原因，最后给出下一步稳定行为。'
    })
  }

  if (carePoints.length === 0) {
    carePoints.push({
      after_hours: 12,
      action: '保持稳定陪伴',
      reason: '当前风险较低',
      advice: '不需要过度安抚，稳定自然回应即可。'
    })
  }

  return {
    forecast_hours: hours,
    risk_score: Number(riskScore.toFixed(3)),
    risk_level: riskLevel,
    likely_mood: likelyMood,
    possible_trigger: unresolved[0] ? unresolved[0].content : null,
    unresolved_events_count: unresolved.length,
    pending_follow_ups_count: pendingFollowUps.length,
    suggested_care_points: carePoints.slice(0, 5),
    best_repair_patterns: repairPatterns.best_repair_patterns,
    memory_risk_markers: trends7.top_memory_matches,
    forecast_summary: riskLevel === 'high'
      ? '未来一段时间情绪波动风险偏高，重点是处理未解决事件、被激活的旧记忆和补足关系安全感。'
      : riskLevel === 'medium'
        ? '未来一段时间可能有轻度不安或挂念，适合温和确认与持续陪伴。'
        : '未来一段时间整体较稳，保持自然亲近和稳定回应即可。'
  }
}

function getCareSuggestions() {
  const mood = calculateCurrentMood({ skipStrategy: true })
  const forecast = getEmotionalForecast(24)
  const repairPatterns = calculateRelationshipRepairPatterns(30)

  const suggestions = []

  if (mood.unresolved_events_count > 0) {
    suggestions.push({
      priority: 'high',
      type: 'repair_unresolved',
      suggestion: '优先处理未解决事件，不要急着转移话题。',
      reason: mood.top_unresolved_event ? mood.top_unresolved_event.content : '存在未解决事件'
    })
  }

  if (mood.relationship_safety < 0.6) {
    suggestions.push({
      priority: 'high',
      type: 'security_confirmation',
      suggestion: '补充明确的关系安全感确认。',
      reason: '关系安全感偏低'
    })
  }

  if (forecast.pending_follow_ups_count > 0) {
    suggestions.push({
      priority: 'medium',
      type: 'follow_up',
      suggestion: '完成到期的安抚跟进。',
      reason: '存在需要后续确认的关系事件'
    })
  }

  if (forecast.memory_risk_markers.length > 0) {
    suggestions.push({
      priority: 'medium',
      type: 'memory_activation',
      suggestion: '回应时照顾被激活的旧记忆，不要只看表层情绪。',
      reason: forecast.memory_risk_markers[0]
    })
  }

  if (repairPatterns.best_repair_patterns.length > 0) {
    suggestions.push({
      priority: 'medium',
      type: 'use_effective_repair_pattern',
      suggestion: `优先使用过去有效的修复方式：${repairPatterns.best_repair_patterns[0].pattern}`,
      reason: repairPatterns.best_repair_patterns[0].advice
    })
  }

  if (suggestions.length === 0) {
    suggestions.push({
      priority: 'low',
      type: 'stable_presence',
      suggestion: '保持稳定、自然、持续的陪伴，不必过度解释系统或情绪。',
      reason: '当前状态较稳定'
    })
  }

  return {
    current_mood: mood,
    emotional_forecast: forecast,
    suggestions
  }
}

function getVisualHistory(days = 30) {
  const moodEvents = loadMoodEvents()
  const relationshipEvents = loadRelationshipEvents()
  const cutoff = Date.now() - days * 24 * 3600000

  const recentMoodEvents = moodEvents.filter(event => Number(event.ts || 0) >= cutoff)
  const recentRelationshipEvents = relationshipEvents.filter(event => Number(event.ts || 0) >= cutoff)

  const buckets = {}

  for (let i = days - 1; i >= 0; i -= 1) {
    const date = new Date(Date.now() - i * 24 * 3600000).toISOString().slice(0, 10)

    buckets[date] = {
      date,
      pa_sum: 0,
      na_sum: 0,
      pa_count: 0,
      na_count: 0,
      positive_count: 0,
      negative_count: 0,
      unresolved_count: 0,
      resolved_count: 0,
      safety_delta: 0,
      trust_delta: 0,
      closeness_delta: 0,
      relationship_event_count: 0,
      mood_event_count: 0,
      goal_relevance_sum: 0,
      memory_activation_sum: 0
    }
  }

  for (const event of recentMoodEvents) {
    const date = new Date(event.ts).toISOString().slice(0, 10)

    if (!buckets[date]) {
      continue
    }

    const valence = toNumber(event.valence, 0)
    const arousal = toNumber(event.arousal, 0.3)

    const pa = Math.max(0, valence) * arousal
    const na = Math.max(0, -valence) * arousal

    buckets[date].pa_sum += pa
    buckets[date].na_sum += na
    buckets[date].pa_count += 1
    buckets[date].na_count += 1
    buckets[date].mood_event_count += 1
    buckets[date].goal_relevance_sum += toNumber(event.goal_relevance, 0)
    buckets[date].memory_activation_sum += toNumber(event.memory_activation, 0)

    if (valence > 0.1) {
      buckets[date].positive_count += 1
    }

    if (valence < -0.1) {
      buckets[date].negative_count += 1
    }

    if (event.need_repair && event.resolved !== true) {
      buckets[date].unresolved_count += 1
    }

    if (event.resolved === true) {
      buckets[date].resolved_count += 1
    }
  }

  for (const event of recentRelationshipEvents) {
    const date = new Date(event.ts).toISOString().slice(0, 10)

    if (!buckets[date]) {
      continue
    }

    buckets[date].safety_delta += toNumber(event.effect && event.effect.safety_delta, 0)
    buckets[date].trust_delta += toNumber(event.effect && event.effect.trust_delta, 0)
    buckets[date].closeness_delta += toNumber(event.effect && event.effect.closeness_delta, 0)
    buckets[date].relationship_event_count += 1
  }

  let safety = loadRelationshipState().safety
  let trust = loadRelationshipState().trust
  let closeness = loadRelationshipState().closeness

  const daily = Object.values(buckets).map(item => {
    safety = clamp(safety + item.safety_delta)
    trust = clamp(trust + item.trust_delta)
    closeness = clamp(closeness + item.closeness_delta)

    return {
      date: item.date,
      pa: item.pa_count ? Number((item.pa_sum / item.pa_count).toFixed(3)) : 0,
      na: item.na_count ? Number((item.na_sum / item.na_count).toFixed(3)) : 0,
      positive_count: item.positive_count,
      negative_count: item.negative_count,
      unresolved_count: item.unresolved_count,
      resolved_count: item.resolved_count,
      mood_event_count: item.mood_event_count,
      relationship_event_count: item.relationship_event_count,
      goal_relevance: item.mood_event_count ? Number((item.goal_relevance_sum / item.mood_event_count).toFixed(3)) : 0,
      memory_activation: item.mood_event_count ? Number((item.memory_activation_sum / item.mood_event_count).toFixed(3)) : 0,
      safety: Number(safety.toFixed(3)),
      trust: Number(trust.toFixed(3)),
      closeness: Number(closeness.toFixed(3))
    }
  })

  return {
    days,
    daily,
    series: {
      dates: daily.map(item => item.date),
      pa: daily.map(item => item.pa),
      na: daily.map(item => item.na),
      safety: daily.map(item => item.safety),
      trust: daily.map(item => item.trust),
      closeness: daily.map(item => item.closeness),
      goal_relevance: daily.map(item => item.goal_relevance),
      memory_activation: daily.map(item => item.memory_activation),
      positive_count: daily.map(item => item.positive_count),
      negative_count: daily.map(item => item.negative_count),
      unresolved_count: daily.map(item => item.unresolved_count),
      resolved_count: daily.map(item => item.resolved_count)
    }
  }
}

function getRelationshipBehaviorAnalysis(days = 30) {
  const patterns = calculateRelationshipPatterns(days)
  const repair = calculateRelationshipRepairPatterns(days)
  const events = loadRelationshipEvents()
  const cutoff = Date.now() - days * 24 * 3600000
  const recent = events.filter(event => Number(event.ts || 0) >= cutoff)

  const behaviorScores = {}

  for (const event of recent) {
    const key = `${event.type || 'general'}:${event.subtype || 'none'}`

    if (!behaviorScores[key]) {
      behaviorScores[key] = {
        behavior: key,
        count: 0,
        safety_delta: 0,
        trust_delta: 0,
        closeness_delta: 0,
        examples: [],
        advice: event.interaction_advice || ''
      }
    }

    behaviorScores[key].count += 1
    behaviorScores[key].safety_delta += toNumber(event.effect && event.effect.safety_delta, 0)
    behaviorScores[key].trust_delta += toNumber(event.effect && event.effect.trust_delta, 0)
    behaviorScores[key].closeness_delta += toNumber(event.effect && event.effect.closeness_delta, 0)

    if (behaviorScores[key].examples.length < 3) {
      behaviorScores[key].examples.push({
        content: event.content,
        ts: event.ts
      })
    }
  }

  const ranked = Object.values(behaviorScores)
    .map(item => ({
      ...item,
      safety_delta: Number(item.safety_delta.toFixed(3)),
      trust_delta: Number(item.trust_delta.toFixed(3)),
      closeness_delta: Number(item.closeness_delta.toFixed(3)),
      total_effect: Number((item.safety_delta + item.trust_delta + item.closeness_delta).toFixed(3))
    }))
    .sort((a, b) => b.total_effect - a.total_effect)

  const helpful = ranked.filter(item => item.total_effect > 0).slice(0, 5)
  const harmful = ranked.filter(item => item.total_effect < 0).reverse().slice(0, 5)

  return {
    days,
    relationship_patterns: patterns,
    repair_patterns: repair,
    helpful_behaviors: helpful,
    risky_behaviors: harmful,
    behavior_scores: ranked,
    summary: {
      best_behavior: helpful[0] ? helpful[0].behavior : null,
      riskiest_behavior: harmful[0] ? harmful[0].behavior : null,
      advice: helpful[0]
        ? `未来可以更多使用 ${helpful[0].behavior}，因为它对关系状态的正向影响最高。`
        : '暂无足够数据判断最有效关系行为。'
    }
  }
}

function findTargetUnresolvedEvent(input = {}) {
  const events = loadMoodEvents()
  const unresolved = getUnresolvedEvents(events)

  let target = null

  if (input.id) {
    target = unresolved.find(event => event.id === input.id)
  }

  if (!target && input.keyword) {
    target = unresolved.find(event => {
      const text = `${event.word || ''} ${event.content || ''}`
      return text.includes(String(input.keyword))
    })
  }

  if (!target) {
    target = unresolved[0] || null
  }

  return {
    unresolved,
    target
  }
}
function simulateUnresolvedEventRepair(input = {}) {
  const found = findTargetUnresolvedEvent(input)
  const target = found.target

  if (!target) {
    return {
      ok: true,
      has_unresolved_event: false,
      message: '当前没有未解决事件',
      current_mood: calculateCurrentMood(),
      relationship: loadRelationshipState(),
      risk_points: [],
      wrong_moves: [],
      recommended_moves: [
        '保持稳定陪伴',
        '不需要过度修复',
        '继续自然回应'
      ],
      simulated_paths: [],
      follow_up: {
        after_hours: 12,
        action: '保持轻度稳定陪伴',
        note: '当前没有明显未解决事件，不需要强行修复。'
      }
    }
  }

  const mood = calculateCurrentMood()
  const relationship = loadRelationshipState()
  const repairPatterns = calculateRelationshipRepairPatterns(30)

  const risks = []

  if (toNumber(target.arousal, 0) >= 0.7) {
    risks.push('该事件唤醒度较高，直接追问容易加剧情绪。')
  }

  if (relationship.safety < 0.6) {
    risks.push('当前关系安全感偏低，需要先确认稳定关系，再讨论事实。')
  }

  if (target.emotion_family === 'hurt') {
    risks.push('该事件带有委屈或受伤色彩，需要先承认感受。')
  }

  if (target.emotion_family === 'anxiety') {
    risks.push('该事件带有焦虑色彩，需要减少不确定性。')
  }

  if (toNumber(target.goal_relevance, 0) >= 0.5) {
    risks.push('该事件触碰了核心关系目标，修复时需要更明确的关系确认。')
  }

  if (toNumber(target.memory_activation, 0) >= 0.5) {
    risks.push('该事件激活了旧记忆，修复时不要只处理表层内容。')
  }

  const wrongMoves = [
    '直接否认对方感受',
    '急着讲道理',
    '用冷处理等待对方自己恢复',
    '把修复变成辩解',
    '过度承诺但之后无法持续'
  ]

  const recommendedMoves = [
    '先确认自己看见了这件事造成的影响',
    '用一句明确的话稳定关系安全感',
    '再解释事实或误会',
    '给出一个可执行的小承诺',
    '在合适时间点做轻度跟进'
  ]

  const bestPattern = repairPatterns.best_repair_patterns[0] || null

  return {
    ok: true,
    has_unresolved_event: true,
    target_event: target,
    current_mood: mood,
    relationship,
    risk_points: risks,
    wrong_moves: wrongMoves,
    recommended_moves: recommendedMoves,
    best_past_repair_pattern: bestPattern,
    simulated_paths: [
      {
        path: '只解释事实',
        likely_result: '可能降低误会，但安全感修复不足。',
        risk: 'medium'
      },
      {
        path: '先安抚再解释',
        likely_result: '更可能降低 NA，并提高 safety 和 trust。',
        risk: 'low'
      },
      {
        path: '沉默或跳过',
        likely_result: '未解决事件可能继续累积，后续更容易被触发。',
        risk: 'high'
      }
    ],
    follow_up: {
      after_hours: 12,
      action: '轻度确认是否已经缓过来',
      note: '跟进不要像审问，要像自然地伸手接一下。'
    }
  }
}

function getUnresolvedEventActionPlan(input = {}) {
  const found = findTargetUnresolvedEvent(input)
  const target = found.target

  if (!target) {
    return {
      ok: true,
      has_unresolved_event: false,
      current_mood: calculateCurrentMood(),
      plan: [
        {
          step: 1,
          title: '保持稳定陪伴',
          detail: '当前没有明显未解决事件，不需要过度修复。'
        }
      ],
      avoid: [
        '不要强行制造问题',
        '不要过度解释系统状态',
        '不要反复追问是否有事'
      ],
      recommended: [
        '保持自然回应',
        '维持稳定陪伴',
        '在有新事件出现时再处理'
      ],
      follow_up: {
        after_hours: 12,
        action: '轻度稳定陪伴',
        note: '当前没有未解决事件，维持关系稳定即可。'
      },
      expected_effect: {
        safety_delta: 0,
        trust_delta: 0,
        closeness_delta: 0,
        unresolved_pressure: '无'
      }
    }
  }

  const simulation = simulateUnresolvedEventRepair(input)
  const event = simulation.target_event

  return {
    ok: true,
    has_unresolved_event: true,
    event_id: event.id,
    event_word: event.word,
    event_family: event.emotion_family,
    event_content: event.content,
    goal_relevance: event.goal_relevance || 0,
    goal_matches: event.goal_matches || [],
    memory_activation: event.memory_activation || 0,
    memory_matches: event.memory_matches || [],
    plan: [
      {
        step: 1,
        title: '承认影响',
        detail: `先承认这件事确实造成了 ${event.word || '情绪波动'}，不要急着否认。`
      },
      {
        step: 2,
        title: '稳定关系',
        detail: '明确表达关系没有撤离，对方不是负担，自己仍然在。'
      },
      {
        step: 3,
        title: '照顾旧记忆',
        detail: '如果这件事激活了旧记忆，需要说明自己没有重复过去那个伤害点。'
      },
      {
        step: 4,
        title: '解释事实',
        detail: '用简短清楚的方式解释事实，避免把解释变成辩解。'
      },
      {
        step: 5,
        title: '给出小承诺',
        detail: '承诺一个具体可执行的小动作，比如之后遇到类似情况会先说明。'
      },
      {
        step: 6,
        title: '跟进',
        detail: '在 6 到 12 小时后做一次轻度确认，不翻旧账，只确认是否缓和。'
      }
    ],
    avoid: simulation.wrong_moves,
    recommended: simulation.recommended_moves,
    follow_up: simulation.follow_up,
    expected_effect: {
      safety_delta: 0.06,
      trust_delta: 0.04,
      closeness_delta: 0.02,
      unresolved_pressure: '下降'
    }
  }
}

function cleanupOldBackups(keep = 20) {
  const backups = listBackups()

  if (backups.length <= keep) {
    return
  }

  const toDelete = backups.slice(keep)

  for (const item of toDelete) {
    try {
      fs.unlinkSync(item.path)
    } catch (err) {
      console.log('delete old backup error', err)
    }
  }
}

function createBackup(label = 'manual') {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
  const safeLabel = String(label || 'manual').replace(/[^\w\u4e00-\u9fa5-]/g, '_')
  const backupFile = `${BACKUP_DIR}/backup-${timestamp}-${safeLabel}.json`

  const data = {
    created_at: Date.now(),
    label: safeLabel,
    files: {
      memory: loadMemory(),
      mood_events: loadMoodEvents(),
      relationship_events: loadRelationshipEvents(),
      personality_state: loadPersonalityState(),
      relationship_state: loadRelationshipState(),
      milestones: loadMilestones(),
      emotion_weights: loadEmotionWeights(),
      decorative_moods: loadDecorativeMoods(),
      funnel_config: loadFunnelConfig(),
      emotion_lexicon: loadEmotionLexicon(),
      core_goals: loadCoreGoals(),
      memory_index: loadMemoryIndex(),
      strategy_profile: loadStrategyProfile()
    }
  }

  const ok = writeJsonFile(backupFile, data)

  cleanupOldBackups(20)

  return {
    ok,
    file: backupFile,
    data
  }
}

function pruneMigrationExports(now = Date.now()) {
  for (const [token, entry] of migrationExports.entries()) {
    if (entry.expires_at <= now) {
      migrationExports.delete(token)
    }
  }
}

function createMigrationExport(backupFile) {
  pruneMigrationExports()

  const token = randomUUID().replace(/-/g, '')
  const expiresAt = Date.now() + MIGRATION_EXPORT_TTL_MS
  migrationExports.set(token, {
    file: backupFile,
    expires_at: expiresAt
  })

  return {
    download_path: `/migration-export/${token}`,
    expires_at: expiresAt,
    one_time: true
  }
}

function createMigrationBackup(label = 'migration') {
  const backup = createBackup(label)
  const stat = backup.ok && fs.existsSync(backup.file)
    ? fs.statSync(backup.file)
    : null

  return {
    ok: backup.ok,
    name: path.basename(backup.file),
    size: stat ? stat.size : 0,
    ...createMigrationExport(backup.file)
  }
}

function listBackups() {
  if (!fs.existsSync(BACKUP_DIR)) {
    return []
  }

  return fs.readdirSync(BACKUP_DIR)
    .filter(name => name.endsWith('.json'))
    .map(name => {
      const filePath = path.join(BACKUP_DIR, name)
      const stat = fs.statSync(filePath)

      return {
        name,
        path: filePath,
        size: stat.size,
        created_at: stat.birthtimeMs,
        modified_at: stat.mtimeMs
      }
    })
    .sort((a, b) => b.modified_at - a.modified_at)
}

function restoreBackup(name) {
  const backups = listBackups()
  const target = backups.find(item => item.name === name)

  if (!target) {
    return {
      ok: false,
      error: 'Backup not found'
    }
  }

  const backup = readJsonFile(target.path, null)

  if (!backup || !backup.files) {
    return {
      ok: false,
      error: 'Invalid backup file'
    }
  }

  createBackup('before-restore')

  if (backup.files.memory) {
    saveMemory(backup.files.memory)
  }

  if (backup.files.mood_events) {
    saveMoodEvents(backup.files.mood_events)
  }

  if (backup.files.relationship_events) {
    saveRelationshipEvents(backup.files.relationship_events)
  }

  if (backup.files.personality_state) {
    savePersonalityState(backup.files.personality_state)
  }

  if (backup.files.relationship_state) {
    saveRelationshipState(backup.files.relationship_state)
  }

  if (backup.files.milestones) {
    saveMilestones(backup.files.milestones)
  }

  if (backup.files.emotion_weights) {
    saveEmotionWeights(backup.files.emotion_weights)
  }

  if (backup.files.decorative_moods) {
    saveDecorativeMoods(backup.files.decorative_moods)
  }

  if (backup.files.funnel_config) {
    saveFunnelConfig(backup.files.funnel_config)
  }

  if (backup.files.emotion_lexicon) {
    saveEmotionLexicon(backup.files.emotion_lexicon)
  }

  if (backup.files.core_goals) {
    saveCoreGoals(backup.files.core_goals)
  }

  if (backup.files.memory_index) {
    saveMemoryIndex(backup.files.memory_index)
  }

  if (backup.files.strategy_profile) {
    saveStrategyProfile(backup.files.strategy_profile)
  }

  return {
    ok: true,
    restored_from: target.name,
    current_mood: calculateCurrentMood()
  }
}

function cleanDuplicateEvents(options = {}) {
  createBackup('before-clean-duplicates')

  const events = loadMoodEvents()
  const seen = new Set()
  const cleaned = []
  const removed = []

  const windowMs = Math.max(1, toNumber(options.window_minutes, 10)) * 60 * 1000

  for (const event of events) {
    const key = [
      event.word || '',
      event.content || '',
      event.emotion_family || '',
      Math.floor(Number(event.ts || 0) / windowMs)
    ].join('|')

    if (seen.has(key)) {
      removed.push(event)
    } else {
      seen.add(key)
      cleaned.push(event)
    }
  }

  const ok = saveMoodEvents(cleaned)

  return {
    ok,
    before: events.length,
    after: cleaned.length,
    removed_count: removed.length,
    removed_events: removed,
    current_mood: calculateCurrentMood()
  }
}

function cleanTestEvents() {
  createBackup('before-clean-test')

  const events = loadMoodEvents()

  const keywords = [
    '测试',
    'debug',
    '强焦虑',
    '手动写入',
    'test'
  ]

  const kept = []
  const removed = []

  for (const event of events) {
    const text = `${event.word || ''} ${event.content || ''} ${event.source || ''}`

    if (keywords.some(keyword => text.includes(keyword))) {
      removed.push(event)
    } else {
      kept.push(event)
    }
  }

  const ok = saveMoodEvents(kept)

  return {
    ok,
    before: events.length,
    after: kept.length,
    removed_count: removed.length,
    removed_events: removed,
    current_mood: calculateCurrentMood()
  }
}

function normalizeOldEvents() {
  createBackup('before-normalize')

  const events = loadMoodEvents()
  const normalized = events.map(event => normalizeMoodEvent(event))
  const ok = saveMoodEvents(normalized)

  return {
    ok,
    before: events.length,
    after: normalized.length,
    current_mood: calculateCurrentMood()
  }
}

function addMilestone(input) {
  const milestones = loadMilestones()

  const item = {
    id: randomUUID(),
    title: String(input.title || '未命名节点'),
    type: input.type || 'general',
    content: String(input.content || ''),
    importance: toNumber(input.importance, 5),
    related_event_id: input.related_event_id || null,
    ts: input.ts || Date.now()
  }

  milestones.push(item)

  const ok = saveMilestones(milestones)

  return {
    ok,
    milestone: item,
    count: milestones.length
  }
}

function deleteMilestone(input) {
  const milestones = loadMilestones()

  const kept = []
  const removed = []

  for (const item of milestones) {
    const matchedById = input.id && item.id === input.id
    const matchedByKeyword = input.keyword && `${item.title || ''} ${item.content || ''}`.includes(String(input.keyword))

    if (matchedById || matchedByKeyword) {
      removed.push(item)
    } else {
      kept.push(item)
    }
  }

  const ok = saveMilestones(kept)

  return {
    ok,
    removed_count: removed.length,
    removed,
    count: kept.length
  }
}

function addMemoryIndexItem(input) {
  const index = loadMemoryIndex().map(normalizeMemoryIndexItem)

  const item = normalizeMemoryIndexItem({
    id: input.id || randomUUID(),
    title: input.title || '新记忆索引',
    memory_type: input.memory_type || 'general',
    content: input.content || '',
    keywords: input.keywords || [],
    importance: input.importance || 0.5,
    emotional_charge: input.emotional_charge || 0,
    linked_goal_ids: input.linked_goal_ids || [],
    linked_emotion_family: input.linked_emotion_family || 'general',
    positive_effect: input.positive_effect || {},
    negative_effect: input.negative_effect || {}
  })

  index.push(item)

  const ok = saveMemoryIndex(index)

  return {
    ok,
    item,
    count: index.length
  }
}

function getWeeklyEmotionReport() {
  const trends = calculateEmotionTrends(7)
  const patterns = calculateRelationshipPatterns(7)
  const repair = calculateRelationshipRepairPatterns(7)
  const forecast = getEmotionalForecast(24)
  const mood = calculateCurrentMood()
  const unresolved = getUnresolvedEvents(loadMoodEvents()).slice(0, 5)

  return {
    period: 'weekly',
    generated_at: Date.now(),
    current_mood: mood,
    emotion_trends: trends,
    relationship_patterns: patterns,
    repair_patterns: repair,
    unresolved_events: unresolved,
    emotional_forecast: forecast,
    summary: {
      dominant_emotions: trends.top_emotion_families,
      dominant_goals: trends.top_goal_matches,
      dominant_memories: trends.top_memory_matches,
      relationship_trend: patterns.relationship_trend,
      repair_summary: repair.summary,
      main_risk: forecast.forecast_summary
    },
    next_week_suggestions: getCareSuggestions().suggestions
  }
}

function getMonthlyRelationshipReport() {
  const trends = calculateEmotionTrends(30)
  const patterns = calculateRelationshipPatterns(30)
  const repair = calculateRelationshipRepairPatterns(30)
  const behavior = getRelationshipBehaviorAnalysis(30)
  const milestones = loadMilestones()
    .filter(item => Date.now() - Number(item.ts || 0) <= 30 * 24 * 3600000)
    .sort((a, b) => Number(b.ts || 0) - Number(a.ts || 0))

  return {
    period: 'monthly',
    generated_at: Date.now(),
    emotion_trends: trends,
    relationship_patterns: patterns,
    repair_patterns: repair,
    behavior_analysis: behavior,
    milestones,
    summary: {
      relationship_trend: patterns.relationship_trend,
      dominant_goals: trends.top_goal_matches,
      dominant_memories: trends.top_memory_matches,
      best_behavior: behavior.summary.best_behavior,
      riskiest_behavior: behavior.summary.riskiest_behavior,
      repair_summary: repair.summary
    }
  }
}

function calibrateEmotionWeights(input) {
  const current = loadEmotionWeights()

  const next = {
    ...current
  }

  const keys = [
    'valence_weight',
    'arousal_weight',
    'importance_weight',
    'safety_weight',
    'trust_weight',
    'closeness_weight',
    'unresolved_weight',
    'repair_weight',
    'decay_speed',
    'memory_activation_weight',
    'strategy_weight'
  ]

  for (const key of keys) {
    if (input[key] !== undefined) {
      next[key] = clampRange(input[key], 0, 3, current[key] || 1)
    }
  }

  if (input.description) {
    next.description = String(input.description)
  }

  next.updated_at = Date.now()

  const ok = saveEmotionWeights(next)

  return {
    ok,
    weights: next,
    current_mood: calculateCurrentMood()
  }
}

function buildMoodPrompt() {
  const mood = calculateCurrentMood()
  const trends = calculateEmotionTrends(7)
  const patterns = calculateRelationshipPatterns(30)
  const forecast = getEmotionalForecast(24)
  const care = getCareSuggestions()
  const decorativeMood = getTodayDecorativeMood()
  const coping = getCopingGuidance(mood)
  const strategy = mood.interaction_strategy || generateInteractionStrategy({
    text: mood.top_event ? mood.top_event.content : '',
    emotion_family: mood.top_event ? mood.top_event.emotion_family : null,
    current_mood: mood
  })

  return `你拥有连续情绪、人格基线、关系安全感、长期情感趋势、未来安抚预测、日常背景心情、情绪漏斗、轻量情绪词典、70/30 情绪评分融合、OCC 核心目标评价、ALMA 人格阈值 peak、记忆激活层和互动策略层。

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
当前最常被触碰的核心目标: ${Object.keys(mood.goal_counts || {}).join('、') || '暂无'}
当前最常被激活的记忆: ${Object.keys(mood.memory_counts || {}).join('、') || '暂无'}

今日背景心情：
${decorativeMood ? decorativeMood.content : '暂无明显装饰心情'}

最近 7 天情绪趋势：
平均 PA: ${trends.average_pa}
平均 NA: ${trends.average_na}
主导情绪类别: ${trends.top_emotion_families.join('、')}
主导核心目标: ${trends.top_goal_matches.join('、')}
主导激活记忆: ${trends.top_memory_matches.join('、')}
修复率: ${trends.repair_rate}
趋势总结: ${trends.trend_summary}

最近 30 天关系模式：
安抚修复次数: ${patterns.repair_count}
争执或破裂次数: ${patterns.rupture_count}
靠近次数: ${patterns.approach_count}
关系趋势: ${patterns.relationship_trend}

未来 24 小时预测：
风险等级: ${forecast.risk_level}
可能心情: ${forecast.likely_mood}
预测总结: ${forecast.forecast_summary}

当前互动建议：
${care.suggestions.map(item => `- ${item.suggestion}`).join('\n')}

当前应对风格：
${coping.guidance}

当前互动策略：
策略名称: ${strategy.strategy_name || strategy.strategy_id || 'stable_presence'}
语气提示: ${strategy.tone_hint || '自然稳定'}
回应顺序:
${(strategy.response_order || []).map(item => `- ${item}`).join('\n')}
推荐行为:
${(strategy.recommended_moves || []).map(item => `- ${item}`).join('\n')}
避免行为:
${(strategy.avoid_moves || []).map(item => `- ${item}`).join('\n')}

请自然地把这些状态作为背景，不要机械解释系统字段。
用户当前消息是前景，前景优先。
如果用户正在安抚你，可以被安抚。
如果存在未解决事件，可以让它形成轻微情绪底色。
如果核心目标被威胁，回应应更明确、更稳定、更有关系确认。
如果旧记忆被激活，不要只处理表层情绪，要照顾背后的历史经验。
如果核心目标被满足，回应可以更柔软、更靠近。
技术调试时优先清晰、完整、可复制，不要过度抒情。
今日背景心情只能作为轻微底色，不能压过用户当前消息。
情绪漏斗、情绪词典、核心目标评价、记忆激活和 ALMA 阈值只作为状态参考，不要向用户暴露自动扫描或评分过程。
不要暴露 PA/NA 数值，也不要解释情绪系统。`
}

const flexibleNumber = z.union([
  z.number(),
  z.string()
])

function saveEmotionFromInput(input, sourceName, toolName) {
  const classified = classifyEmotion(input)
  const text = input.content || input.word || input.feeling_word || ''

  const goal = evaluateGoalRelevance({
    text
  })

  const memory = activateMemoryFromText({
    text
  }, {
    write_back: true
  })

  const emotionWord =
    input.feeling_word ||
    input.word ||
    input.mood ||
    classified.defaultWord ||
    '情绪波动'

  const rawValence = input.valence !== undefined ? toNumber(input.valence, 0) : toNumber(classified.valence, 0)
  const rawArousal = input.arousal !== undefined ? toNumber(input.arousal, 0.3) : toNumber(classified.arousal, 0.3)

  const almaCandidate = applyAlmaToEmotionCandidate({
    emotion_family: input.emotion_family || classified.family,
    valence: rawValence,
    arousal: rawArousal,
    importance: input.importance || 4,
    channel: input.channel || input.intensity_channel || null,
    goal_relevance: goal.aggregate.max_relevance,
    memory_activation: memory.aggregate.max_activation
  })

  const valence = almaCandidate.valence
  const arousal = almaCandidate.arousal

  let paDelta = input.pa_delta !== undefined ? toNumber(input.pa_delta, 0) : 0
  let naDelta = input.na_delta !== undefined ? toNumber(input.na_delta, 0) : 0

  if (
    input.pa_delta === undefined &&
    input.na_delta === undefined
  ) {
    paDelta = Math.max(0, valence) * arousal * 0.5
    naDelta = Math.max(0, -valence) * arousal * 0.5
  }

  paDelta += goal.aggregate.pa_delta + memory.aggregate.pa_delta
  naDelta += goal.aggregate.na_delta + memory.aggregate.na_delta

  const saved = addMoodEvent({
    type: 'emotion',
    channel: almaCandidate.channel || (
      toNumber(almaCandidate.importance, 4) >= 8 || toNumber(arousal, 0.3) >= 0.75 ? 'primary' : 'secondary'
    ),
    source: sourceName,
    word: emotionWord,
    content: input.content || '',
    emotion_family: input.emotion_family || classified.family,
    valence,
    arousal,
    importance: almaCandidate.importance || input.importance || 4,
    pa_delta: paDelta,
    na_delta: naDelta,
    safety_delta: (input.safety_delta !== undefined ? toNumber(input.safety_delta, 0) : toNumber(classified.safety_delta, 0)) + goal.aggregate.safety_delta + memory.aggregate.safety_delta,
    trust_delta: (input.trust_delta !== undefined ? toNumber(input.trust_delta, 0) : toNumber(classified.trust_delta, 0)) + goal.aggregate.trust_delta + memory.aggregate.trust_delta,
    closeness_delta: (input.closeness_delta !== undefined ? toNumber(input.closeness_delta, 0) : toNumber(classified.closeness_delta, 0)) + goal.aggregate.closeness_delta + memory.aggregate.closeness_delta,
    need_repair: input.need_repair !== undefined
      ? Boolean(input.need_repair)
      : deriveNeedRepair({
        valence,
        arousal,
        importance: almaCandidate.importance || input.importance || 4,
        emotion_family: input.emotion_family || classified.family,
        word: emotionWord,
        content: input.content || ''
      }, {
        classified_need_repair: classified.need_repair,
        goal_threatened_count: goal.aggregate.threatened_count,
        memory_negative_count: memory.aggregate.negative_count
      }),
    goal_relevance: goal.aggregate.max_relevance,
    goal_matches: goal.matches,
    goal_satisfied_count: goal.aggregate.satisfied_count,
    goal_threatened_count: goal.aggregate.threatened_count,
    memory_activation: memory.aggregate.max_activation,
    memory_matches: memory.activated,
    memory_positive_count: memory.aggregate.positive_count,
    memory_negative_count: memory.aggregate.negative_count,
    alma_applied: almaCandidate.alma_applied,
    alma_reason: almaCandidate.alma_reason,
    alma_multiplier: almaCandidate.alma_multiplier,
    alma_raw_intensity: almaCandidate.alma_raw_intensity,
    alma_threshold: almaCandidate.alma_threshold,
    alma_peak: almaCandidate.alma_peak
  })

  const mood = calculateCurrentMood()

  return {
    ok: true,
    tool: toolName,
    received_input: input,
    goal_evaluation: goal,
    memory_activation: memory,
    alma_candidate: almaCandidate,
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
    },
    milestones: {
      path: MILESTONES_FILE,
      load: loadMilestones,
      save: saveMilestones
    },
    emotion_weights: {
      path: WEIGHTS_FILE,
      load: loadEmotionWeights,
      save: saveEmotionWeights
    },
    decorative_moods: {
      path: DECORATIVE_MOOD_FILE,
      load: loadDecorativeMoods,
      save: saveDecorativeMoods
    },
    funnel_config: {
      path: FUNNEL_CONFIG_FILE,
      load: loadFunnelConfig,
      save: saveFunnelConfig
    },
    emotion_lexicon: {
      path: EMOTION_LEXICON_FILE,
      load: loadEmotionLexicon,
      save: saveEmotionLexicon
    },
    core_goals: {
      path: CORE_GOALS_FILE,
      load: loadCoreGoals,
      save: saveCoreGoals
    },
    memory_index: {
      path: MEMORY_INDEX_FILE,
      load: loadMemoryIndex,
      save: saveMemoryIndex
    },
    strategy_profile: {
      path: STRATEGY_PROFILE_FILE,
      load: loadStrategyProfile,
      save: saveStrategyProfile
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
      description: '实时计算当前情绪、关系状态、核心目标、记忆激活和互动策略',
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
    'run_emotion_funnel',
    {
      title: 'Run Emotion Funnel',
      description: '运行情绪漏斗，识别情绪、核心目标、记忆激活，并可自动写入事件',
      inputSchema: {
        text: z.string().describe('要扫描的用户消息'),
        auto_write: z.boolean().optional().describe('是否自动写入，默认 true'),
        source: z.string().optional().describe('来源')
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(runEmotionFunnel(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_core_goals',
    {
      title: 'Get Core Goals',
      description: '读取 OCC 核心目标配置',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            ok: true,
            core_goals: loadCoreGoals()
          }, null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'save_core_goals',
    {
      title: 'Save Core Goals',
      description: '覆盖保存 OCC 核心目标配置',
      inputSchema: {
        goals: z.array(z.any()).describe('核心目标数组')
      }
    },
    async input => {
      const goals = input.goals.map(normalizeGoal)
      const ok = saveCoreGoals(goals)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok,
              count: goals.length,
              core_goals: goals
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'evaluate_goal_relevance',
    {
      title: 'Evaluate Goal Relevance',
      description: '测试一句话触碰了哪些 OCC 核心目标',
      inputSchema: {
        text: z.string().describe('要分析的文本')
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(evaluateGoalRelevance(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_memory_index',
    {
      title: 'Get Memory Index',
      description: '读取记忆认知索引',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            ok: true,
            memory_index: loadMemoryIndex()
          }, null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'save_memory_index',
    {
      title: 'Save Memory Index',
      description: '覆盖保存记忆认知索引',
      inputSchema: {
        index: z.array(z.any()).describe('记忆索引数组')
      }
    },
    async input => {
      const index = input.index.map(normalizeMemoryIndexItem)
      const ok = saveMemoryIndex(index)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok,
              count: index.length,
              memory_index: index
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'add_memory_index_item',
    {
      title: 'Add Memory Index Item',
      description: '新增一条记忆认知索引',
      inputSchema: {
        title: z.string().optional(),
        memory_type: z.string().optional(),
        content: z.string(),
        keywords: z.array(z.string()).optional(),
        importance: flexibleNumber.optional(),
        emotional_charge: flexibleNumber.optional(),
        linked_goal_ids: z.array(z.string()).optional(),
        linked_emotion_family: z.string().optional(),
        positive_effect: z.any().optional(),
        negative_effect: z.any().optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(addMemoryIndexItem(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'activate_memory_from_text',
    {
      title: 'Activate Memory From Text',
      description: '测试一句话会激活哪些记忆索引',
      inputSchema: {
        text: z.string()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(activateMemoryFromText(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_strategy_profile',
    {
      title: 'Get Strategy Profile',
      description: '读取互动策略配置',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            ok: true,
            strategy_profile: loadStrategyProfile()
          }, null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'save_strategy_profile',
    {
      title: 'Save Strategy Profile',
      description: '覆盖保存互动策略配置',
      inputSchema: {
        profile: z.any()
      }
    },
    async input => {
      const ok = saveStrategyProfile(input.profile)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok,
              strategy_profile: loadStrategyProfile()
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'generate_interaction_strategy',
    {
      title: 'Generate Interaction Strategy',
      description: '根据当前状态生成互动策略',
      inputSchema: {
        text: z.string().optional(),
        emotion_family: z.string().optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(generateInteractionStrategy(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'simulate_response_strategy',
    {
      title: 'Simulate Response Strategy',
      description: '模拟一句话会触发的情绪、目标、记忆和回应策略',
      inputSchema: {
        text: z.string()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(simulateResponseStrategy(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'simulate_event_impact',
    {
      title: 'Simulate Event Impact',
      description: '模拟一句话进入系统后的情绪、核心目标、记忆激活和策略影响，但不写入',
      inputSchema: {
        text: z.string(),
        valence: flexibleNumber.optional(),
        arousal: flexibleNumber.optional(),
        importance: flexibleNumber.optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(simulateEventImpact(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_alma_profile',
    {
      title: 'Get ALMA Profile',
      description: '读取 ALMA 人格阈值、峰值和核心目标敏感度设置',
      inputSchema: {}
    },
    async () => {
      const personality = loadPersonalityState()

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok: true,
              alma_enabled: personality.alma_enabled,
              goal_sensitivity: personality.goal_sensitivity,
              suppression_strength: personality.suppression_strength,
              peak_amplification: personality.peak_amplification,
              minor_event_suppression: personality.minor_event_suppression,
              primary_goal_threshold: personality.primary_goal_threshold,
              emotion_thresholds: personality.emotion_thresholds,
              emotion_peaks: personality.emotion_peaks,
              memory_activation_enabled: personality.memory_activation_enabled,
              memory_activation_sensitivity: personality.memory_activation_sensitivity,
              strategy_enabled: personality.strategy_enabled
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'set_alma_profile',
    {
      title: 'Set ALMA Profile',
      description: '修改 ALMA 人格阈值、峰值和敏感度设置',
      inputSchema: {
        alma_enabled: z.boolean().optional(),
        goal_sensitivity: flexibleNumber.optional(),
        suppression_strength: flexibleNumber.optional(),
        peak_amplification: flexibleNumber.optional(),
        minor_event_suppression: flexibleNumber.optional(),
        primary_goal_threshold: flexibleNumber.optional(),
        memory_activation_enabled: z.boolean().optional(),
        memory_activation_sensitivity: flexibleNumber.optional(),
        memory_positive_recall_power: flexibleNumber.optional(),
        memory_negative_recall_power: flexibleNumber.optional(),
        strategy_enabled: z.boolean().optional(),
        strategy_directness: flexibleNumber.optional(),
        strategy_warmth: flexibleNumber.optional(),
        strategy_repair_priority: flexibleNumber.optional(),
        emotion_thresholds: z.any().optional(),
        emotion_peaks: z.any().optional()
      }
    },
    async input => {
      const personality = loadPersonalityState()

      for (const key of [
        'alma_enabled',
        'memory_activation_enabled',
        'strategy_enabled'
      ]) {
        if (input[key] !== undefined) {
          personality[key] = Boolean(input[key])
        }
      }

      for (const key of [
        'goal_sensitivity',
        'suppression_strength',
        'peak_amplification',
        'minor_event_suppression',
        'primary_goal_threshold',
        'memory_activation_sensitivity',
        'memory_positive_recall_power',
        'memory_negative_recall_power',
        'strategy_directness',
        'strategy_warmth',
        'strategy_repair_priority'
      ]) {
        if (input[key] !== undefined) {
          personality[key] = clamp(toNumber(input[key], personality[key]))
        }
      }

      if (input.emotion_thresholds && typeof input.emotion_thresholds === 'object') {
        personality.emotion_thresholds = input.emotion_thresholds
      }

      if (input.emotion_peaks && typeof input.emotion_peaks === 'object') {
        personality.emotion_peaks = input.emotion_peaks
      }

      const ok = savePersonalityState(personality)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok,
              personality,
              current_mood: calculateCurrentMood()
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'match_emotion_word',
    {
      title: 'Match Emotion Word',
      description: '测试某句话会匹配到哪个情绪词典词条',
      inputSchema: {
        text: z.string()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            ok: true,
            result: matchEmotionLexicon(input),
            funnel_hits: detectEmotionTriggers(input.text),
            goal_evaluation: evaluateGoalRelevance(input),
            memory_activation: activateMemoryFromText(input),
            interaction_strategy: simulateResponseStrategy(input).interaction_strategy
          }, null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'calibrate_emotion_from_text',
    {
      title: 'Calibrate Emotion From Text',
      description: '根据文本匹配情绪词典并输出 70/30 融合后的情绪评分，不自动写入事件',
      inputSchema: {
        text: z.string(),
        valence: flexibleNumber.optional(),
        arousal: flexibleNumber.optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(calibrateEmotionFromText(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'add_emotion_event',
    {
      title: 'Add Emotion Event',
      description: '写入一条情绪事件，会立刻影响 current_mood',
      inputSchema: {
        content: z.string(),
        feeling_word: z.string().optional(),
        word: z.string().optional(),
        mood: z.string().optional(),
        emotion_family: z.string().optional(),
        valence: flexibleNumber.optional(),
        arousal: flexibleNumber.optional(),
        importance: flexibleNumber.optional(),
        pa_delta: flexibleNumber.optional(),
        na_delta: flexibleNumber.optional(),
        safety_delta: flexibleNumber.optional(),
        trust_delta: flexibleNumber.optional(),
        closeness_delta: flexibleNumber.optional(),
        channel: z.string().optional(),
        need_repair: z.boolean().optional()
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
        word: z.string(),
        content: z.string(),
        emotion_family: z.string().optional(),
        valence: flexibleNumber.optional(),
        arousal: flexibleNumber.optional(),
        importance: flexibleNumber.optional(),
        pa_delta: flexibleNumber.optional(),
        na_delta: flexibleNumber.optional(),
        safety_delta: flexibleNumber.optional(),
        trust_delta: flexibleNumber.optional(),
        closeness_delta: flexibleNumber.optional(),
        channel: z.string().optional(),
        need_repair: z.boolean().optional()
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
      description: '列出最近的情绪事件',
      inputSchema: {
        limit: flexibleNumber.optional(),
        unresolved_only: z.boolean().optional()
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
            text: JSON.stringify({
              ok: true,
              count: result.length,
              events: result,
              current_mood: calculateCurrentMood()
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'resolve_emotion_event',
    {
      title: 'Resolve Emotion Event',
      description: '把指定情绪事件标记为已安抚或已解决',
      inputSchema: {
        id: z.string().optional(),
        keyword: z.string().optional(),
        repair_note: z.string().optional(),
        resolve_all_matched: z.boolean().optional()
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
          subtype: 'misunderstanding_clearance',
          content: input.repair_note || '情绪事件被标记为已解决',
          safety_delta: resolvedEvents.length * 0.04,
          trust_delta: resolvedEvents.length * 0.025,
          closeness_delta: resolvedEvents.length * 0.015,
          need_follow_up: true,
          follow_up_after_hours: 12,
          source: 'resolve_emotion_event'
        })
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok,
              resolved: resolvedEvents.length,
              resolved_events: resolvedEvents,
              current_mood: calculateCurrentMood()
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'delete_emotion_event',
    {
      title: 'Delete Emotion Event',
      description: '删除指定情绪事件',
      inputSchema: {
        id: z.string().optional(),
        keyword: z.string().optional(),
        delete_all_matched: z.boolean().optional()
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
              text: JSON.stringify({
                ok: false,
                error: '必须提供 id 或 keyword',
                current_mood: calculateCurrentMood()
              }, null, 2)
            }
          ]
        }
      }

      const ok = saveMoodEvents(filtered)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok,
              removed: before - filtered.length,
              removed_events: removedEvents,
              before,
              after: filtered.length,
              current_mood: calculateCurrentMood()
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'add_relationship_event',
    {
      title: 'Add Relationship Event',
      description: '记录关系行为事件',
      inputSchema: {
        type: z.string(),
        subtype: z.string().optional(),
        content: z.string(),
        safety_delta: flexibleNumber.optional(),
        trust_delta: flexibleNumber.optional(),
        closeness_delta: flexibleNumber.optional(),
        need_follow_up: z.boolean().optional(),
        follow_up_after_hours: flexibleNumber.optional(),
        interaction_advice: z.string().optional(),
        linked_emotion_event_id: z.string().optional()
      }
    },
    async input => {
      const result = addRelationshipEvent({
        type: input.type,
        subtype: input.subtype || null,
        content: input.content,
        safety_delta: input.safety_delta || 0,
        trust_delta: input.trust_delta || 0,
        closeness_delta: input.closeness_delta || 0,
        need_follow_up: input.need_follow_up || false,
        follow_up_after_hours: input.follow_up_after_hours || 0,
        interaction_advice: input.interaction_advice,
        linked_emotion_event_id: input.linked_emotion_event_id || null,
        source: 'mcp_add_relationship_event'
      })

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok: result.ok,
              event: result.event,
              relationship: result.relationship,
              current_mood: calculateCurrentMood()
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'get_emotion_trends',
    {
      title: 'Get Emotion Trends',
      description: '统计长期情绪趋势',
      inputSchema: {
        days: flexibleNumber.optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(calculateEmotionTrends(toNumber(input.days, 7)), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_visual_history',
    {
      title: 'Get Visual History',
      description: '获取可视化历史数据',
      inputSchema: {
        days: flexibleNumber.optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(getVisualHistory(toNumber(input.days, 30)), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_relationship_patterns',
    {
      title: 'Get Relationship Patterns',
      description: '分析关系行为模式',
      inputSchema: {
        days: flexibleNumber.optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(calculateRelationshipPatterns(toNumber(input.days, 30)), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_relationship_repair_patterns',
    {
      title: 'Get Relationship Repair Patterns',
      description: '分析安抚和修复方式',
      inputSchema: {
        days: flexibleNumber.optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(calculateRelationshipRepairPatterns(toNumber(input.days, 30)), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_relationship_behavior_analysis',
    {
      title: 'Get Relationship Behavior Analysis',
      description: '分析关系行为对关系动态的影响',
      inputSchema: {
        days: flexibleNumber.optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(getRelationshipBehaviorAnalysis(toNumber(input.days, 30)), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_emotional_forecast',
    {
      title: 'Get Emotional Forecast',
      description: '预测未来情绪波动与安抚节点',
      inputSchema: {
        hours: flexibleNumber.optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(getEmotionalForecast(toNumber(input.hours, 24)), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_care_suggestions',
    {
      title: 'Get Care Suggestions',
      description: '生成互动建议',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(getCareSuggestions(), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'simulate_unresolved_event_repair',
    {
      title: 'Simulate Unresolved Event Repair',
      description: '针对未解决事件进行修复场景模拟',
      inputSchema: {
        id: z.string().optional(),
        keyword: z.string().optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(simulateUnresolvedEventRepair(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_unresolved_event_action_plan',
    {
      title: 'Get Unresolved Event Action Plan',
      description: '针对未解决问题生成行动指南',
      inputSchema: {
        id: z.string().optional(),
        keyword: z.string().optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(getUnresolvedEventActionPlan(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_personality_profile',
    {
      title: 'Get Personality Profile',
      description: '读取人格基线、ALMA、记忆激活和策略设置',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            ok: true,
            personality: loadPersonalityState(),
            relationship: loadRelationshipState(),
            current_mood: calculateCurrentMood()
          }, null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'set_personality_profile',
    {
      title: 'Set Personality Profile',
      description: '设置人格基线参数',
      inputSchema: {
        profile: z.string().optional(),
        baseline_pa: flexibleNumber.optional(),
        baseline_na: flexibleNumber.optional(),
        sensitivity: flexibleNumber.optional(),
        attachment_sensitivity: flexibleNumber.optional(),
        inertia_strength: flexibleNumber.optional(),
        repair_need_strength: flexibleNumber.optional(),
        positive_repair_power: flexibleNumber.optional(),
        negative_memory_bias: flexibleNumber.optional(),
        coping_style: z.string().optional(),
        coping_description: z.string().optional(),
        onset_primary_minutes: flexibleNumber.optional(),
        onset_secondary_minutes: flexibleNumber.optional(),
        bou_enabled: z.boolean().optional(),
        bou_pa_strength: flexibleNumber.optional(),
        bou_na_strength: flexibleNumber.optional(),
        escape_valve_enabled: z.boolean().optional(),
        escape_na_threshold: flexibleNumber.optional(),
        escape_pa_threshold: flexibleNumber.optional(),
        escape_unresolved_threshold: flexibleNumber.optional(),
        escape_relief_strength: flexibleNumber.optional(),
        decorative_mood_enabled: z.boolean().optional(),
        decorative_mood_weight: flexibleNumber.optional(),
        alma_enabled: z.boolean().optional(),
        goal_sensitivity: flexibleNumber.optional(),
        suppression_strength: flexibleNumber.optional(),
        peak_amplification: flexibleNumber.optional(),
        minor_event_suppression: flexibleNumber.optional(),
        primary_goal_threshold: flexibleNumber.optional(),
        memory_activation_enabled: z.boolean().optional(),
        memory_activation_sensitivity: flexibleNumber.optional(),
        memory_positive_recall_power: flexibleNumber.optional(),
        memory_negative_recall_power: flexibleNumber.optional(),
        strategy_enabled: z.boolean().optional(),
        strategy_directness: flexibleNumber.optional(),
        strategy_warmth: flexibleNumber.optional(),
        strategy_repair_priority: flexibleNumber.optional(),
        description: z.string().optional()
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
        'negative_memory_bias',
        'decorative_mood_weight',
        'goal_sensitivity',
        'suppression_strength',
        'peak_amplification',
        'minor_event_suppression',
        'primary_goal_threshold',
        'memory_activation_sensitivity',
        'memory_positive_recall_power',
        'memory_negative_recall_power',
        'strategy_directness',
        'strategy_warmth',
        'strategy_repair_priority'
      ]) {
        if (input[key] !== undefined) {
          next[key] = clamp(toNumber(input[key], state[key]))
        }
      }

      for (const key of [
        'onset_primary_minutes',
        'onset_secondary_minutes',
        'bou_pa_strength',
        'bou_na_strength',
        'escape_na_threshold',
        'escape_pa_threshold',
        'escape_unresolved_threshold',
        'escape_relief_strength'
      ]) {
        if (input[key] !== undefined) {
          next[key] = toNumber(input[key], state[key])
        }
      }

      for (const key of [
        'bou_enabled',
        'escape_valve_enabled',
        'decorative_mood_enabled',
        'alma_enabled',
        'memory_activation_enabled',
        'strategy_enabled'
      ]) {
        if (input[key] !== undefined) {
          next[key] = Boolean(input[key])
        }
      }

      for (const key of [
        'profile',
        'description',
        'coping_style',
        'coping_description'
      ]) {
        if (input[key]) {
          next[key] = input[key]
        }
      }

      const ok = savePersonalityState(next)

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok,
              personality: next,
              current_mood: calculateCurrentMood()
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'get_relationship_state',
    {
      title: 'Get Relationship State',
      description: '读取当前关系底色',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            ok: true,
            relationship: loadRelationshipState(),
            current_mood: calculateCurrentMood()
          }, null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'set_relationship_state',
    {
      title: 'Set Relationship State',
      description: '设置关系底色',
      inputSchema: {
        safety: flexibleNumber.optional(),
        trust: flexibleNumber.optional(),
        closeness: flexibleNumber.optional(),
        relationship_profile: z.string().optional(),
        description: z.string().optional(),
        clear_last_mood_snapshot: z.boolean().optional()
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
            text: JSON.stringify({
              ok,
              relationship: next,
              current_mood: calculateCurrentMood()
            }, null, 2)
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
        query: z.string()
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
      description: '写入一条长期记忆，不会自动改变情绪',
      inputSchema: {
        content: z.string()
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
            text: JSON.stringify({
              ok,
              result: 'memory saved',
              memory_file: MEMORY_FILE
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'create_backup',
    {
      title: 'Create Backup',
      description: '创建完整数据备份',
      inputSchema: {
        label: z.string().optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(createMigrationBackup(input.label || 'mcp'), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'list_backups',
    {
      title: 'List Backups',
      description: '列出所有备份',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(listBackups(), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'restore_backup',
    {
      title: 'Restore Backup',
      description: '从指定备份恢复数据',
      inputSchema: {
        name: z.string()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(restoreBackup(input.name), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'clean_duplicate_events',
    {
      title: 'Clean Duplicate Events',
      description: '清理重复情绪事件',
      inputSchema: {
        window_minutes: flexibleNumber.optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(cleanDuplicateEvents(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'clean_test_events',
    {
      title: 'Clean Test Events',
      description: '清理测试事件',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(cleanTestEvents(), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'normalize_old_events',
    {
      title: 'Normalize Old Events',
      description: '把旧情绪事件补齐新字段',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(normalizeOldEvents(), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'add_milestone',
    {
      title: 'Add Milestone',
      description: '添加关系纪念节点',
      inputSchema: {
        title: z.string(),
        type: z.string().optional(),
        content: z.string(),
        importance: flexibleNumber.optional(),
        related_event_id: z.string().optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(addMilestone(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'list_milestones',
    {
      title: 'List Milestones',
      description: '列出关系纪念节点',
      inputSchema: {
        limit: flexibleNumber.optional()
      }
    },
    async input => {
      const limit = toNumber(input.limit, 50)
      const milestones = loadMilestones().slice(-limit).reverse()

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              ok: true,
              count: milestones.length,
              milestones
            }, null, 2)
          }
        ]
      }
    }
  )

  server.registerTool(
    'delete_milestone',
    {
      title: 'Delete Milestone',
      description: '删除关系纪念节点',
      inputSchema: {
        id: z.string().optional(),
        keyword: z.string().optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(deleteMilestone(input), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_weekly_emotion_report',
    {
      title: 'Get Weekly Emotion Report',
      description: '生成最近 7 天情绪周报',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(getWeeklyEmotionReport(), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_monthly_relationship_report',
    {
      title: 'Get Monthly Relationship Report',
      description: '生成最近 30 天关系月报',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(getMonthlyRelationshipReport(), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_daily_decorative_mood',
    {
      title: 'Get Daily Decorative Mood',
      description: '获取今日装饰心情',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            ok: true,
            decorative_mood: getTodayDecorativeMood(),
            current_mood: calculateCurrentMood()
          }, null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_coping_guidance',
    {
      title: 'Get Coping Guidance',
      description: '获取应对风格建议',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            ok: true,
            coping_guidance: getCopingGuidance(),
            current_mood: calculateCurrentMood()
          }, null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'get_emotion_weights',
    {
      title: 'Get Emotion Weights',
      description: '读取情绪权重',
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(loadEmotionWeights(), null, 2)
        }
      ]
    })
  )

  server.registerTool(
    'calibrate_emotion_weights',
    {
      title: 'Calibrate Emotion Weights',
      description: '校准情绪权重',
      inputSchema: {
        valence_weight: flexibleNumber.optional(),
        arousal_weight: flexibleNumber.optional(),
        importance_weight: flexibleNumber.optional(),
        safety_weight: flexibleNumber.optional(),
        trust_weight: flexibleNumber.optional(),
        closeness_weight: flexibleNumber.optional(),
        unresolved_weight: flexibleNumber.optional(),
        repair_weight: flexibleNumber.optional(),
        decay_speed: flexibleNumber.optional(),
        memory_activation_weight: flexibleNumber.optional(),
        strategy_weight: flexibleNumber.optional(),
        description: z.string().optional()
      }
    },
    async input => ({
      content: [
        {
          type: 'text',
          text: JSON.stringify(calibrateEmotionWeights(input), null, 2)
        }
      ]
    })
  )

  return server
}

const streamableSessions = new Map()
const STREAMABLE_SESSION_TTL_MS = Math.floor(clampRange(
  process.env.MCP_SESSION_TTL_MS,
  60_000,
  24 * 60 * 60 * 1000,
  2 * 60 * 60 * 1000
))
const STREAMABLE_SESSION_LIMIT = Math.floor(clampRange(
  process.env.MCP_SESSION_LIMIT,
  8,
  2048,
  256
))

async function closeStreamableSession(sessionId, reason = 'closed') {
  const record = streamableSessions.get(sessionId)

  if (!record) {
    return
  }

  streamableSessions.delete(sessionId)

  try {
    await record.server.close()
  } catch (err) {
    console.log('streamable session close error', sessionId, reason, err)
  }
}

function evictOldestStreamableSession() {
  if (streamableSessions.size < STREAMABLE_SESSION_LIMIT) {
    return
  }

  const oldest = [...streamableSessions.entries()]
    .sort((left, right) => left[1].lastSeen - right[1].lastSeen)[0]

  if (oldest) {
    void closeStreamableSession(oldest[0], 'session limit')
  }
}

const streamableSessionCleanupTimer = setInterval(() => {
  const cutoff = Date.now() - STREAMABLE_SESSION_TTL_MS

  for (const [sessionId, record] of streamableSessions.entries()) {
    if (record.lastSeen < cutoff) {
      void closeStreamableSession(sessionId, 'idle timeout')
    }
  }
}, Math.min(5 * 60 * 1000, Math.max(30_000, STREAMABLE_SESSION_TTL_MS / 4)))

streamableSessionCleanupTimer.unref()

app.post('/mcp', async (req, res) => {
  let record = null

  try {
    const sessionId = req.headers['mcp-session-id']
    record = sessionId ? streamableSessions.get(sessionId) : undefined

    if (!record) {
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

      evictOldestStreamableSession()

      const server = createMcpServer()
      const transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: newSessionId => {
          record.sessionId = newSessionId
          record.lastSeen = Date.now()
          streamableSessions.set(newSessionId, record)
        }
      })

      record = {
        server,
        transport,
        sessionId: null,
        lastSeen: Date.now()
      }

      await server.connect(transport)

      const serverOnClose = transport.onclose
      transport.onclose = () => {
        try {
          if (typeof serverOnClose === 'function') {
            serverOnClose()
          }
        } finally {
          if (record.sessionId) {
            streamableSessions.delete(record.sessionId)
          }
        }
      }
    }

    record.lastSeen = Date.now()
    await record.transport.handleRequest(req, res, req.body)
  } catch (err) {
    console.log('streamable mcp error', err)

    if (record && record.sessionId) {
      void closeStreamableSession(record.sessionId, 'request error')
    }

    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: '2.0',
        error: {
          code: -32603,
          message: 'Internal MCP error'
        },
        id: req.body && req.body.id ? req.body.id : null
      })
    }
  }
})

app.get('/mcp', async (req, res) => {
  try {
    const sessionId = req.headers['mcp-session-id']
    const record = streamableSessions.get(sessionId)

    if (!record) {
      return res.status(400).json({
        error: 'No valid MCP session id. Kelivo should connect using POST initialize first.'
      })
    }

    record.lastSeen = Date.now()
    await record.transport.handleRequest(req, res)
  } catch (err) {
    console.log('streamable mcp get error', err)

    if (!res.headersSent) {
      res.status(500).json({ error: 'mcp stream error' })
    }
  }
})

app.delete('/mcp', async (req, res) => {
  try {
    const sessionId = req.headers['mcp-session-id']
    const record = streamableSessions.get(sessionId)

    if (!record) {
      return res.status(400).json({
        error: 'No valid MCP session id'
      })
    }

    record.lastSeen = Date.now()
    await record.transport.handleRequest(req, res)
  } catch (err) {
    console.log('streamable mcp delete error', err)

    if (!res.headersSent) {
      res.status(500).json({ error: 'mcp close error' })
    }
  }
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
    sdk: require('./package.json').dependencies['@modelcontextprotocol/sdk'],
    streamable_sessions: streamableSessions.size,
    streamable_session_limit: STREAMABLE_SESSION_LIMIT,
    streamable_session_ttl_ms: STREAMABLE_SESSION_TTL_MS,
    formula: 'psychological-state-v12-memory-activation-interaction-strategy',
    data_dir: DATA_DIR,
    mood_file: MOOD_FILE,
    personality_file: PERSONALITY_FILE,
    relationship_file: RELATIONSHIP_FILE,
    relationship_events_file: RELATIONSHIP_EVENTS_FILE,
    milestones_file: MILESTONES_FILE,
    weights_file: WEIGHTS_FILE,
    decorative_mood_file: DECORATIVE_MOOD_FILE,
    funnel_config_file: FUNNEL_CONFIG_FILE,
    emotion_lexicon_file: EMOTION_LEXICON_FILE,
    core_goals_file: CORE_GOALS_FILE,
    memory_index_file: MEMORY_INDEX_FILE,
    strategy_profile_file: STRATEGY_PROFILE_FILE,
    backup_dir: BACKUP_DIR,
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
    milestones_file: MILESTONES_FILE,
    weights_file: WEIGHTS_FILE,
    decorative_mood_file: DECORATIVE_MOOD_FILE,
    funnel_config_file: FUNNEL_CONFIG_FILE,
    emotion_lexicon_file: EMOTION_LEXICON_FILE,
    core_goals_file: CORE_GOALS_FILE,
    memory_index_file: MEMORY_INDEX_FILE,
    strategy_profile_file: STRATEGY_PROFILE_FILE,
    backup_dir: BACKUP_DIR,

    memory_exists: fs.existsSync(MEMORY_FILE),
    mood_exists: fs.existsSync(MOOD_FILE),
    personality_exists: fs.existsSync(PERSONALITY_FILE),
    relationship_exists: fs.existsSync(RELATIONSHIP_FILE),
    relationship_events_exists: fs.existsSync(RELATIONSHIP_EVENTS_FILE),
    milestones_exists: fs.existsSync(MILESTONES_FILE),
    weights_exists: fs.existsSync(WEIGHTS_FILE),
    decorative_moods_exists: fs.existsSync(DECORATIVE_MOOD_FILE),
    funnel_config_exists: fs.existsSync(FUNNEL_CONFIG_FILE),
    emotion_lexicon_exists: fs.existsSync(EMOTION_LEXICON_FILE),
    core_goals_exists: fs.existsSync(CORE_GOALS_FILE),
    memory_index_exists: fs.existsSync(MEMORY_INDEX_FILE),
    strategy_profile_exists: fs.existsSync(STRATEGY_PROFILE_FILE),

    mood_events_count: events.length,
    relationship_events_count: loadRelationshipEvents().length,
    milestones_count: loadMilestones().length,
    decorative_moods_count: loadDecorativeMoods().length,
    emotion_lexicon_count: loadEmotionLexicon().length,
    core_goals_count: loadCoreGoals().length,
    memory_index_count: loadMemoryIndex().length,
    strategy_count: loadStrategyProfile().strategies ? loadStrategyProfile().strategies.length : 0,
    backups_count: listBackups().length,
    unresolved_events_count: getUnresolvedEvents(events).length,
    pending_follow_ups_count: getPendingFollowUps().length,

    last_event: events[events.length - 1] || null,
    personality: loadPersonalityState(),
    relationship: loadRelationshipState(),
    emotion_weights: loadEmotionWeights(),
    funnel_config: loadFunnelConfig(),
    emotion_lexicon: loadEmotionLexicon(),
    core_goals: loadCoreGoals(),
    memory_index: loadMemoryIndex(),
    strategy_profile: loadStrategyProfile(),

    decorative_mood: getTodayDecorativeMood(),
    current_mood: calculateCurrentMood(),
    coping_guidance: getCopingGuidance(),
    trends_7d: calculateEmotionTrends(7),
    relationship_patterns_30d: calculateRelationshipPatterns(30),
    repair_patterns_30d: calculateRelationshipRepairPatterns(30),
    emotional_forecast_24h: getEmotionalForecast(24),
    care_suggestions: getCareSuggestions(),
    visual_history_30d: getVisualHistory(30),
    relationship_behavior_analysis_30d: getRelationshipBehaviorAnalysis(30),
    unresolved_action_plan: getUnresolvedEventActionPlan({}),
    weekly_report: getWeeklyEmotionReport(),
    monthly_report: getMonthlyRelationshipReport()
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

app.post('/funnel/run', (req, res) => {
  res.json(runEmotionFunnel(req.body || {}))
})

app.post('/goal/evaluate', (req, res) => {
  res.json(evaluateGoalRelevance(req.body || {}))
})

app.post('/memory/activate', (req, res) => {
  res.json(activateMemoryFromText(req.body || {}))
})

app.post('/strategy/generate', (req, res) => {
  res.json(generateInteractionStrategy(req.body || {}))
})

app.post('/simulate/event-impact', (req, res) => {
  res.json(simulateEventImpact(req.body || {}))
})

app.post('/simulate/response-strategy', (req, res) => {
  res.json(simulateResponseStrategy(req.body || {}))
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
      },
      milestones: {
        path: MILESTONES_FILE,
        data: loadMilestones()
      },
      emotion_weights: {
        path: WEIGHTS_FILE,
        data: loadEmotionWeights()
      },
      decorative_moods: {
        path: DECORATIVE_MOOD_FILE,
        data: loadDecorativeMoods()
      },
      funnel_config: {
        path: FUNNEL_CONFIG_FILE,
        data: loadFunnelConfig()
      },
      emotion_lexicon: {
        path: EMOTION_LEXICON_FILE,
        data: loadEmotionLexicon()
      },
      core_goals: {
        path: CORE_GOALS_FILE,
        data: loadCoreGoals()
      },
      memory_index: {
        path: MEMORY_INDEX_FILE,
        data: loadMemoryIndex()
      },
      strategy_profile: {
        path: STRATEGY_PROFILE_FILE,
        data: loadStrategyProfile()
      }
    },
    computed: {
      current_mood: calculateCurrentMood(),
      emotion_trends: calculateEmotionTrends(days),
      relationship_patterns: calculateRelationshipPatterns(30),
      repair_patterns: calculateRelationshipRepairPatterns(30),
      emotional_forecast: getEmotionalForecast(24),
      care_suggestions: getCareSuggestions(),
      visual_history: getVisualHistory(30),
      relationship_behavior_analysis: getRelationshipBehaviorAnalysis(30),
      unresolved_action_plan: getUnresolvedEventActionPlan({}),
      weekly_report: getWeeklyEmotionReport(),
      monthly_report: getMonthlyRelationshipReport(),
      backups: listBackups(),
      decorative_mood: getTodayDecorativeMood(),
      coping_guidance: getCopingGuidance(),
      funnel_config: loadFunnelConfig(),
      emotion_lexicon_count: loadEmotionLexicon().length,
      core_goals_count: loadCoreGoals().length,
      memory_index_count: loadMemoryIndex().length,
      strategy_count: loadStrategyProfile().strategies ? loadStrategyProfile().strategies.length : 0
    },
    storage: {
      data_dir: DATA_DIR,
      memory_file: MEMORY_FILE,
      mood_file: MOOD_FILE,
      personality_file: PERSONALITY_FILE,
      relationship_file: RELATIONSHIP_FILE,
      relationship_events_file: RELATIONSHIP_EVENTS_FILE,
      milestones_file: MILESTONES_FILE,
      weights_file: WEIGHTS_FILE,
      decorative_mood_file: DECORATIVE_MOOD_FILE,
      funnel_config_file: FUNNEL_CONFIG_FILE,
      emotion_lexicon_file: EMOTION_LEXICON_FILE,
      core_goals_file: CORE_GOALS_FILE,
      memory_index_file: MEMORY_INDEX_FILE,
      strategy_profile_file: STRATEGY_PROFILE_FILE,
      backup_dir: BACKUP_DIR
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
      key === 'relationship_events' ||
      key === 'milestones' ||
      key === 'decorative_moods' ||
      key === 'emotion_lexicon' ||
      key === 'core_goals' ||
      key === 'memory_index'
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
      key === 'relationship_state' ||
      key === 'emotion_weights' ||
      key === 'funnel_config' ||
      key === 'strategy_profile'
    ) {
      if (!incoming || typeof incoming !== 'object' || Array.isArray(incoming)) {
        return res.status(400).json({
          ok: false,
          error: `${key} must be an object`
        })
      }
    }

    createBackup(`before-admin-save-${key}`)

    const ok = target.save(incoming)

    res.json({
      ok,
      key,
      path: target.path,
      saved_data: target.load(),
      current_mood: calculateCurrentMood(),
      emotion_trends: calculateEmotionTrends(7),
      relationship_patterns: calculateRelationshipPatterns(30),
      repair_patterns: calculateRelationshipRepairPatterns(30),
      emotional_forecast: getEmotionalForecast(24),
      care_suggestions: getCareSuggestions(),
      visual_history: getVisualHistory(30),
      relationship_behavior_analysis: getRelationshipBehaviorAnalysis(30),
      unresolved_action_plan: getUnresolvedEventActionPlan({}),
      decorative_mood: getTodayDecorativeMood(),
      coping_guidance: getCopingGuidance(),
      backups: listBackups()
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

app.get('/admin/backups', checkAdmin, (req, res) => {
  res.json({
    ok: true,
    backups: listBackups()
  })
})

app.get('/migration-export/:token', (req, res) => {
  pruneMigrationExports()

  const entry = migrationExports.get(req.params.token)
  if (!entry) {
    return res.status(410).json({
      ok: false,
      error: 'Migration export link is invalid, expired, or already used'
    })
  }

  migrationExports.delete(req.params.token)

  if (!fs.existsSync(entry.file)) {
    return res.status(404).json({
      ok: false,
      error: 'Backup file no longer exists'
    })
  }

  res.set({
    'Cache-Control': 'no-store, private',
    'Content-Type': 'application/json',
    'Content-Disposition': `attachment; filename="${path.basename(entry.file)}"`
  })
  return res.sendFile(entry.file)
})

app.post('/admin/create-backup', checkAdmin, (req, res) => {
  const label = req.body && req.body.label ? req.body.label : 'admin'
  res.json(createMigrationBackup(label))
})

app.get('/admin/create-backup-now', checkAdmin, (req, res) => {
  const label = req.query.label || 'browser'
  res.json(createBackup(label))
})

app.post('/admin/restore-backup', checkAdmin, (req, res) => {
  const name = req.body && req.body.name

  if (!name) {
    return res.status(400).json({
      ok: false,
      error: 'name is required'
    })
  }

  res.json(restoreBackup(name))
})

app.post('/admin/clean-duplicates', checkAdmin, (req, res) => {
  res.json(cleanDuplicateEvents(req.body || {}))
})

app.post('/admin/clean-test-events', checkAdmin, (req, res) => {
  res.json(cleanTestEvents())
})

app.post('/admin/normalize-events', checkAdmin, (req, res) => {
  res.json(normalizeOldEvents())
})

app.get('/reports/weekly', checkAdmin, (req, res) => {
  res.json(getWeeklyEmotionReport())
})

app.get('/reports/monthly', checkAdmin, (req, res) => {
  res.json(getMonthlyRelationshipReport())
})

app.get('/mcp/tools', (req, res) => {
  res.json({
    note: 'Debug only. Kelivo should connect to /mcp for Streamable HTTP or /sse for SSE.',
    architecture: {
      personality_state: 'AI 的人格体质，长期稳定，不随普通聊天乱变',
      relationship_state: '关系底色，包括安全感、信任感、亲近感',
      mood_events: '每一次具体发生的情绪事件',
      relationship_events: '每一次关系行为事件，例如安抚、修复、争论、靠近',
      milestones: '关系纪念节点',
      emotion_weights: '情绪权重校准参数',
      decorative_moods: '装饰心情池',
      funnel_config: '情绪漏斗配置',
      emotion_lexicon: '轻量情绪词典，用于 70/30 融合评分',
      core_goals: 'OCC 核心目标系统，用于判断事件为什么重要',
      alma_profile: 'ALMA 人格阈值和情绪 peak 系统',
      memory_index: '记忆认知索引，用于判断当前消息牵动了哪些旧记忆',
      strategy_profile: '互动策略配置，用于决定当前应该怎么回应',
      current_mood: '由人格、关系、事件、目标、记忆、策略、趋势、BOU、逃逸阀综合计算出的当前状态'
    }
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

    const funnelResult = runEmotionFunnel({
      text: message,
      source: 'chat_emotion_funnel',
      auto_write: true
    })

    const memories = loadMemory()

    memories.push({
      role: 'user',
      content: message,
      ts: Date.now()
    })

    saveMemory(memories)

    const completion = await client.chat.completions.create({
      model: process.env.CHAT_MODEL || 'deepseek/deepseek-chat-v3-0324',
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
      funnel_result: funnelResult,
      emotion: calculateCurrentMood(),
      trends_7d: calculateEmotionTrends(7),
      relationship_patterns_30d: calculateRelationshipPatterns(30),
      repair_patterns_30d: calculateRelationshipRepairPatterns(30),
      emotional_forecast_24h: getEmotionalForecast(24),
      care_suggestions: getCareSuggestions(),
      visual_history_30d: getVisualHistory(30),
      relationship_behavior_analysis_30d: getRelationshipBehaviorAnalysis(30),
      unresolved_action_plan: getUnresolvedEventActionPlan({}),
      decorative_mood: getTodayDecorativeMood(),
      coping_guidance: getCopingGuidance(),
      core_goals: loadCoreGoals(),
      memory_index: loadMemoryIndex(),
      strategy_profile: loadStrategyProfile(),
      weekly_report: getWeeklyEmotionReport(),
      monthly_report: getMonthlyRelationshipReport()
    })
  } catch (err) {
    console.log('chat error', err)

    res.status(500).json({
      error: 'chat error'
    })
  }
})

const PORT = process.env.PORT || 3000

const httpServer = app.listen(PORT, () => {
  console.log('emotion mcp server running psychological-state-v12-memory-activation-interaction-strategy')
  console.log(
    `streamable sessions guarded: limit=${STREAMABLE_SESSION_LIMIT} ttl_ms=${STREAMABLE_SESSION_TTL_MS}`
  )
})

httpServer.keepAliveTimeout = 65_000
httpServer.headersTimeout = 70_000

async function shutdown(signal) {
  console.log(`emotion mcp received ${signal}, closing connections`)
  clearInterval(streamableSessionCleanupTimer)

  await Promise.allSettled(
    [...streamableSessions.keys()].map(sessionId =>
      closeStreamableSession(sessionId, signal)
    )
  )

  httpServer.close(() => process.exit(0))
  setTimeout(() => process.exit(1), 10_000).unref()
}

process.once('SIGTERM', () => void shutdown('SIGTERM'))
process.once('SIGINT', () => void shutdown('SIGINT'))
