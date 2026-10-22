FROM node:22-bookworm-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

COPY index.js ./
COPY public ./public

RUN mkdir -p /data && chown -R node:node /app /data

ENV NODE_ENV=production \
    DATA_DIR=/data \
    PORT=3000

USER node

EXPOSE 3000

CMD ["node", "index.js"]
