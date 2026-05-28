const fs = require('fs')

const file = 'index.js'

if (!fs.existsSync(file)) {
  console.error('index.js not found')
  process.exit(1)
}

let code = fs.readFileSync(file, 'utf8')

if (code.includes('const negativeDominant = naFinal >= paFinal - 0.08')) {
  console.log('mood label logic already patched')
  process.exit(0)
}

const oldBlockRegex = /let mood = ['"]平静['"][\s\S]*?if\s*\(\s*paFinal\s*<\s*0\.36\s*&&\s*naFinal\s*<\s*0\.32\s*\)\s*\{\s*mood\s*=\s*['"]安静['"]\s*\}/

const newBlock = `let mood = '平静'

  const negativeDominant = naFinal >= paFinal - 0.08
  const positiveDominant = paFinal >= naFinal + 0.18

  if (paFinal < 0.34 && naFinal < 0.3) {
    mood = '安静'
  }

  if (paFinal >= 0.56 && naFinal < 0.34 && positiveDominant) {
    mood = '开心'
  }

  if (paFinal >= 0.45 && naFinal >= 0.42 && Math.abs(paFinal - naFinal) < 0.18) {
    mood = '复杂'
  }

  if (naFinal >= 0.38 && naFinal < 0.5) {
    mood = '不安'
  }

  if (naFinal >= 0.48 && negativeDominant) {
    mood = '低落'
  }

  if (naFinal >= 0.56 && negativeDominant) {
    mood = '悲伤'
  }

  if (naFinal >= 0.66) {
    mood = '强烈悲伤'
  }

  if (naFinal >= 0.58 && paFinal < 0.42) {
    mood = '痛苦'
  }

  if (
    naFinal >= 0.48 &&
    naFinal > paFinal + 0.12 &&
    topEvent &&
    topEvent.emotion_family === 'anxiety'
  ) {
    mood = '焦虑'
  }

  if (unresolvedEvents.length > 0 && naFinal >= 0.38 && mood === '开心') {
    mood = '挂念'
  }`

if (!oldBlockRegex.test(code)) {
  console.error('没有匹配到旧 mood 判断块。')
  console.error('请检查 index.js 是否是 v12 完整版本，或者把 calculateCurrentMood 那一段发给先生。')
  process.exit(1)
}

code = code.replace(oldBlockRegex, newBlock)

fs.writeFileSync(file, code)

console.log('mood label logic patched successfully')
