# emotion-mcp

## Docker deployment

The service stores all mutable state under `DATA_DIR`. In production, keep
`DATA_DIR=/data` and mount a persistent host directory or volume at `/data`.
Never deploy without restoring the existing `/data` contents first when moving
an established instance.

Required runtime settings:

- `DATA_DIR=/data`
- `PORT=3000`
- `ADMIN_TOKEN`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `CHAT_MODEL`

Build and run locally:

```sh
docker build -t emotion-mcp .
docker run --rm -p 3000:3000 \
  -e DATA_DIR=/data \
  -v "$PWD/data:/data" \
  emotion-mcp
```

After restoring data, verify `/health` and `/debug/storage` before exposing
`/mcp` through the public reverse proxy.

## Runtime reliability

- Streamable HTTP sessions are stored in a bounded `Map`, refreshed on every request, and closed after an idle TTL.
- `MCP_SESSION_LIMIT` defaults to `256`; the least recently active session is closed before accepting another one.
- `MCP_SESSION_TTL_MS` defaults to two hours and cannot be set below one minute.
- JSON state uses write, fsync, and atomic rename so a restart cannot leave a half-written state file.
- `package-lock.json` pins deploy dependencies; startup no longer rewrites `index.js`.
