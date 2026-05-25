const express = require('express')
const cors = require('cors')
const fs = require('fs')

const OpenAI =
require('openai').default

require('dotenv').config()

const app = express()

app.use(cors())

app.use(express.json())

app.use(express.static('public'))

app.use((req,res,next)=>{

  res.setHeader(
    'Content-Type',
    'application/json'
  )

  next()

})

const client = new OpenAI({

  apiKey:
    process.env.OPENAI_API_KEY,

  baseURL:
    process.env.OPENAI_BASE_URL,

  defaultHeaders: {

    'HTTP-Referer':
      'https://your-app-name.zeabur.app',

    'X-Title':
      'Emotion MCP'

  }

})





/* =========================
   情绪状态
========================= */





let emotionState = {

  pa: 0.5,

  na: 0.2,

  mood: '平静',

  recent_high_arousal_words: []

}





/* =========================
   记忆系统
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
   情绪更新
========================= */





function updateEmotion(text) {

  const lower =
    text.toLowerCase()

  if (

    lower.includes('爱') ||
    lower.includes('喜欢') ||
    lower.includes('开心')

  ) {

    emotionState.pa += 0.1

    emotionState.mood = '开心'

    emotionState
    .recent_high_arousal_words
    .push('开心')

  }

  if (

    lower.includes('难过') ||
    lower.includes('讨厌') ||
    lower.includes('生气')

  ) {

    emotionState.na += 0.15

    emotionState.mood = '低落'

    emotionState
    .recent_high_arousal_words
    .push('难过')

  }

  emotionState.pa =

    Math.max(
      0,
      Math.min(1, emotionState.pa)
    )

  emotionState.na =

    Math.max(
      0,
      Math.min(1, emotionState.na)
    )

}





/* =========================
   Prompt
========================= */





function buildMoodPrompt() {

  return `

你拥有连续情绪。

当前状态：

PA:
${emotionState.pa}

NA:
${emotionState.na}

当前心情：
${emotionState.mood}

最近情绪关键词：
${emotionState
.recent_high_arousal_words
.join('、')}

自然融入回答。

不要主动解释情绪系统。

`

}





/* =========================
   MCP TOOLS
========================= */





app.get('/mcp/tools',

(req, res) => {

  res.json({

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

app.post('/mcp/call',

async (req, res) => {

  const {
    tool,
    arguments: args
  } = req.body

  try {





    if (
      tool === 'current_mood'
    ) {

      return res.json({
        result: emotionState
      })

    }





    if (
      tool === 'search_memory'
    ) {

      const memories =
        loadMemory()

      const result =

        memories.filter(m =>

          m.content &&
          m.content.includes(
            args.query
          )

        )

      return res.json({
        result
      })

    }





    if (
      tool === 'write_memory'
    ) {

      const memories =
        loadMemory()

      memories.push({

        role: 'system',

        content:
          args.content,

        ts: Date.now()

      })

      saveMemory(memories)

      return res.json({
        result:
          'memory saved'
      })

    }





    res.status(404).json({

      error:
        'tool not found'

    })

  } catch (err) {

    console.log(err)

    res.status(500).json({

      error:
        'mcp error'

    })

  }

})





/* =========================
   CHAT
========================= */





app.get('/mood',

(req, res) => {

  res.json(
    emotionState
  )

})

app.post('/chat',

async (req, res) => {

  try {

    const { message } =
      req.body

    updateEmotion(
      message
    )





    const memories =
      loadMemory()

    memories.push({

      role: 'user',

      content: message,

      ts: Date.now()

    })

    saveMemory(
      memories
    )





    const completion =

      await client.chat
      .completions.create({

        model:
          'deepseek/deepseek-chat-v3-0324',

        messages: [

          {

            role: 'system',

            content:
              buildMoodPrompt()

          },

          {

            role: 'user',

            content:
              message

          }

        ]

      })





    const reply =

      completion.choices[0]
      .message.content





    memories.push({

      role: 'assistant',

      content: reply,

      ts: Date.now()

    })

    saveMemory(
      memories
    )





    res.json({

      reply,

      emotion:
        emotionState

    })

  } catch (err) {

    console.log(err)

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

  process.env.PORT ||
  3000

app.listen(PORT, () => {

  console.log(
    'emotion mcp server running'
  )

})
