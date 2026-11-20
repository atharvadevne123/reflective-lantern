'use strict';

const fs = require('fs');
const path = require('path');

/**
 * Read and return the primary system prompt markdown.
 *
 * @returns {string} Contents of prompts/system_prompt.md
 * @throws {Error} If the file cannot be read
 */
function getSystemPrompt() {
  const promptPath = path.join(__dirname, 'prompts', 'system_prompt.md');
  try {
    return fs.readFileSync(promptPath, 'utf8');
  } catch (err) {
    throw new Error(`Failed to read system prompt at ${promptPath}: ${err.message}`);
  }
}

/**
 * Load all markdown files in the prompts/ directory.
 *
 * @returns {Record<string, string>} Map of prompt name (without .md) to file contents
 */
function getPrompts() {
  const dir = path.join(__dirname, 'prompts');
  try {
    return fs.readdirSync(dir)
      .filter(f => f.endsWith('.md'))
      .reduce((acc, f) => {
        try {
          acc[path.basename(f, '.md')] = fs.readFileSync(path.join(dir, f), 'utf8');
        } catch (err) {
          console.error(`Warning: could not read prompt file ${f}: ${err.message}`);
        }
        return acc;
      }, {});
  } catch (err) {
    throw new Error(`Failed to read prompts directory at ${dir}: ${err.message}`);
  }
}

/**
 * Load all run history JSON files from the history/ directory.
 *
 * @returns {Record<string, Array|object>} Map of repo name to history entries
 */
function getHistory() {
  const dir = path.join(__dirname, 'history');
  const result = {};
  let files;
  try {
    files = fs.readdirSync(dir).filter(f => f.endsWith('.json') && f !== 'schema.json');
  } catch (err) {
    throw new Error(`Failed to read history directory at ${dir}: ${err.message}`);
  }
  for (const f of files) {
    const repoName = path.basename(f, '.json');
    try {
      const raw = fs.readFileSync(path.join(dir, f), 'utf8');
      result[repoName] = JSON.parse(raw);
    } catch (err) {
      console.error(`Warning: could not parse history file ${f}: ${err.message}`);
    }
  }
  return result;
}

/**
 * Validate that required files exist and are readable.
 *
 * @returns {{ valid: boolean, errors: string[] }}
 */
function validate() {
  const errors = [];
  const required = [
    path.join(__dirname, 'prompts', 'system_prompt.md'),
    path.join(__dirname, 'history'),
    path.join(__dirname, '.env.example'),
  ];
  for (const p of required) {
    if (!fs.existsSync(p)) {
      errors.push(`Missing required path: ${p}`);
    }
  }
  return { valid: errors.length === 0, errors };
}

module.exports = { getSystemPrompt, getPrompts, getHistory, validate };
