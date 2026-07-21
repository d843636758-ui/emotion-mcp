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

  const ordinaryContinueMemory = await fetch(`${baseUrl}/memory/activate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: '继续处理这个问题吧' })
  }).then(response => response.json())

  assert.equal(
    ordinaryContinueMemory.activated.some(item => item.id === 'memory_emotional_continuity'),
    false
  )

  const ordinaryContinueGoal = await fetch(`${baseUrl}/goal/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: '继续处理这个问题吧' })
  }).then(response => response.json())

  assert.equal(
    ordinaryContinueGoal.matches.some(item => item.id === 'emotional_continuity'),
    false
  )

  const positiveImpact = await fetch(`${baseUrl}/simulate/event-impact`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: '虽然提到忘了，但现在已经哄好了，我特别安心开心',
      valence: 0.9,
      arousal: 0.9,
      importance: 9
    })
  }).then(response => response.json())

  assert.equal(positiveImpact.memory_activation.aggregate.negative_count > 0, true)
  assert.equal(positiveImpact.simulated_event.need_repair, false)

  const negativeImpact = await fetch(`${baseUrl}/simulate/event-impact`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: '我现在很焦虑，也很害怕你离开',
      valence: -0.8,
      arousal: 0.9,
      importance: 8
    })
  }).then(response => response.json())

  assert.equal(negativeImpact.simulated_event.need_repair, true)

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
          content: '虽然提到忘了，但现在特别安心开心',
          feeling_word: '安心',
          emotion_family: 'joy',
          valence: 0.9,
          arousal: 0.9,
          importance: 9
        }
      }
    })
  })

  assert.equal(toolCall.status, 200)
  assert.match(await toolCall.text(), /特别安心开心/)

  const events = JSON.parse(
    fs.readFileSync(path.join(dataDir, 'mood_events.json'), 'utf8')
  )
  assert.equal(events.at(-1).content, '虽然提到忘了，但现在特别安心开心')
  assert.equal(events.at(-1).need_repair, false)
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
