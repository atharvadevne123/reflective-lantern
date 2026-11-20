const fs = require('fs');
const path = require('path');

function getSystemPrompt() {
  return fs.readFileSync(path.join(__dirname, 'prompts', 'system_prompt.md'), 'utf8');
}

function getPrompts() {
  const dir = path.join(__dirname, 'prompts');
  return fs.readdirSync(dir)
    .filter(f => f.endsWith('.md'))
    .reduce((acc, f) => {
      acc[path.basename(f, '.md')] = fs.readFileSync(path.join(dir, f), 'utf8');
      return acc;
    }, {});
}

module.exports = { getSystemPrompt, getPrompts };
