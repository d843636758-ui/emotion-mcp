const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { spawn } = require('node:child_process')
const test = require('node:test')


async function waitFor(check, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs
  let lastError = null

  while (Date.now() < deadline) {
    try {
      return await check()
    } catch (error) {
      lastError = error
      await new Promise(resolve => setTimeout(resolve, 100))
    }
  }

  throw lastError || new Error('timed out')
}


test('streamable MCP sessions initialize, write atomically, and close cleanly', async t => {
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'emotion-mcp-test-'))
  const port = 30_000 + (process.pid % 10_000)
  const baseUrl = `http://127.0.0.1:${port}`
  const child = spawn(process.execPath, ['index.js'], {
    cwd: path.resolve(__dirname, '..'),
    env: {
      ...process.env,
      PORT: String(port),
      DATA_DIR: dataDir,
      OPENAI_API_KEY: 'test-key',
      MCP_SESSION_TTL_MS: '60000',
      MCP_SESSION_LIMIT: '8'
    },
    stdio: ['ignore', 'pipe', 'pipe']
  })

  let logs = ''
  child.stdout.on('data', chunk => { logs += chunk })
  child.stderr.on('data', chunk => { logs += chunk })

  t.after(() => {
    child.kill('SIGTERM')
    fs.rmSync(dataDir, { recursive: true, force: true })
  })

  await waitFor(async () => {
    const response = await fetch(`${baseUrl}/health`)
    assert.equal(response.status, 200)
  }).catch(error => {
    throw new Error(`${error.message}\n${logs}`)
  })

  const initialize = await fetch(`${baseUrl}/mcp`, {
    method: 'POST',
    headers: {
      Accept: 'application/json, text/event-stream',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2025-03-26',
        capabilities: {},
        clientInfo: { name: 'integration-test', version: '1' }
      }
    })
  })

  assert.equal(initialize.status, 200)
  const sessionId = initialize.headers.get('mcp-session-id')
  assert.ok(sessionId)

  const afterInitialize = await fetch(`${baseUrl}/health`).then(response => response.json())
  assert.equal(afterInitialize.streamable_sessions, 1)
  assert.equal(afterInitialize.streamable_session_limit, 8)

  const toolCall = await fetch(`${baseUrl}/mcp`, {
    method: 'POST',
    headers: {
      Accept: 'application/json, text/event-stream',
      'Content-Type': 'application/json',
      'Mcp-Session-Id': sessionId
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/call',
      params: {
        name: 'add_emotion_event',
        arguments: {
          content: 'integration test event',
          feeling_word: '安心',
          valence: 0.7,
          arousal: 0.3,
          importance: 4
        }
      }
    })
  })

  assert.equal(toolCall.status, 200)
  assert.match(await toolCall.text(), /integration test event/)

  const events = JSON.parse(
    fs.readFileSync(path.join(dataDir, 'mood_events.json'), 'utf8')
  )
  assert.equal(events.at(-1).content, 'integration test event')
  assert.equal(
    fs.readdirSync(dataDir).some(name => name.endsWith('.tmp')),
    false
  )

  const close = await fetch(`${baseUrl}/mcp`, {
    method: 'DELETE',
    headers: {
      Accept: 'application/json, text/event-stream',
      'Mcp-Session-Id': sessionId
    }
  })

  assert.equal(close.status, 200)
  await waitFor(async () => {
    const health = await fetch(`${baseUrl}/health`).then(response => response.json())
    assert.equal(health.streamable_sessions, 0)
  })

  const sessionIds = []

  for (let index = 0; index < 9; index += 1) {
    const response = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: {
        Accept: 'application/json, text/event-stream',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 10 + index,
        method: 'initialize',
        params: {
          protocolVersion: '2025-03-26',
          capabilities: {},
          clientInfo: { name: 'limit-test', version: '1' }
        }
      })
    })

    assert.equal(response.status, 200)
    sessionIds.push(response.headers.get('mcp-session-id'))
  }

  const atLimit = await fetch(`${baseUrl}/health`).then(response => response.json())
  assert.equal(atLimit.streamable_sessions, 8)

  const evicted = await fetch(`${baseUrl}/mcp`, {
    method: 'POST',
    headers: {
      Accept: 'application/json, text/event-stream',
      'Content-Type': 'application/json',
      'Mcp-Session-Id': sessionIds[0]
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: 30,
      method: 'tools/list',
      params: {}
    })
  })

  assert.equal(evicted.status, 400)

  await Promise.all(
    sessionIds.slice(1).map(activeSessionId =>
      fetch(`${baseUrl}/mcp`, {
        method: 'DELETE',
        headers: {
          Accept: 'application/json, text/event-stream',
          'Mcp-Session-Id': activeSessionId
        }
      })
    )
  )
})
