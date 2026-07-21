# emotion-mcp

## Runtime reliability

- Streamable HTTP sessions are stored in a bounded `Map`, refreshed on every request, and closed after an idle TTL.
- `MCP_SESSION_LIMIT` defaults to `256`; the least recently active session is closed before accepting another one.
- `MCP_SESSION_TTL_MS` defaults to two hours and cannot be set below one minute.
- JSON state uses write, fsync, and atomic rename so a restart cannot leave a half-written state file.
- `package-lock.json` pins deploy dependencies; startup no longer rewrites `index.js`.
