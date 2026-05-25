const express = require('express')
const cors = require('cors')
const fs = require('fs')

const OpenAI =
require('openai').default

require('dotenv').config()

const {
  McpServer
} = require(
  '@modelcontextprotocol/sdk/server/mcp.js'
)

const {
  StreamableHTTPServerTransport
} = require(
  '@modelcontextprotocol/sdk/server/streamableHttp.js'
)

const app = express()

app.use(cors())

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
   EMOTION
========================= */





let emotionState = {

  pa: 0.5,

  na: 0.2,

  mood: '平静'

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

    console.log(err)

  }

}





/* =========================
   MCP SERVER
========================= */





const mcpServer =
new McpServer({

  name:
    'emotion-system',

  version:
    '1.0.0'

})





/* =========================
   TOOL 1
========================= */





mcpServer.tool(

  'current_mood',

  {

    description:
      '获取当前情绪状态',

    inputSchema: {

      type: 'object',

      properties: {},

      required: []

    }

  },

  async () => {

    return {

      content: [

        {

          type: 'text',

          text: JSON.stringify(
            emotionState,
            null,
            2
          )

        }

      ]

    }

  }

)





/* =========================
   TOOL 2
========================= */





mcpServer.tool(

  'search_memory',

  {

    description:
      '搜索历史记忆',

    inputSchema: {

      type: 'object',

      properties: {

        query: {

          type: 'string'

        }

      },

      required: ['query']

    }

  },

  async ({ query }) => {

    const memories =
      loadMemory()

    const result =

      memories.filter(m =>

        m.content &&
        m.content.includes(
          query
        )

      )

    return {

      content: [

        {

          type: 'text',

          text: JSON.stringify(
            result,
            null,
            2
          )

        }

      ]

    }

  }

)





/* =========================
   TOOL 3
========================= */





mcpServer.tool(

  'write_memory',

  {

    description:
      '写入记忆',

    inputSchema: {

      type: 'object',

      properties: {

        content: {

          type: 'string'

        }

      },

      required: ['content']

    }

  },

  async ({ content }) => {

    const memories =
      loadMemory()

    memories.push({

      role: 'system',

      content,

      ts: Date.now()

    })

    saveMemory(memories)

    return {

      content: [

        {

          type: 'text',

          text:
            'memory saved'

        }

      ]

    }

  }

)





/* =========================
   MCP ENDPOINT
========================= */





app.post('/mcp',

async (req, res) => {

  try {

    const transport =

      new StreamableHTTPServerTransport()

    await mcpServer.connect(
      transport
    )

    await transport.handleRequest(
      req,
      res,
      req.body
    )

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





function updateEmotion(text) {

  const lower =
    text.toLowerCase()

  if (
    lower.includes('开心')
  ) {

    emotionState.pa += 0.1

    emotionState.mood =
      '开心'

  }

  if (
    lower.includes('难过')
  ) {

    emotionState.na += 0.15

    emotionState.mood =
      '低落'

  }

}





app.post('/chat',

async (req, res) => {

  try {

    const { message } =
      req.body

    updateEmotion(
      message
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
              `当前情绪：
${emotionState.mood}`

          },

          {

            role: 'user',

            content:
              message

          }

        ]

      })

    res.json({

      reply:

        completion.choices[0]
        .message.content

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
    'official mcp server running'
  )

})
