import assert from 'node:assert/strict';
import {readFile, readdir, stat} from 'node:fs/promises';
import {extname, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build, staticPages} from './build.mjs';
import {splitRow} from './catalog.mjs';

const root = resolve(fileURLToPath(new URL('../', import.meta.url)));
const problems = [];
const report = (file, line, message) => problems.push(`${file}:${line}: ${message}`);
const decodeEntities = value => value.replace(/&(?:amp|quot|apos|lt|gt|nbsp);/g, entity => ({'&amp;': '&', '&quot;': '"', '&apos;': "'", '&lt;': '<', '&gt;': '>', '&nbsp;': ' '})[entity]);

function withoutFences(source) {
  let fence = null;
  return source.split(/\r?\n/).map(line => {
    const marker = line.match(/^ {0,3}(`{3,}|~{3,})/);
    if (fence) {
      if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length && !line.slice(marker[0].length).trim()) fence = null;
      return '';
    }
    if (marker) { fence = marker[1]; return ''; }
    return line;
  }).join('\n');
}

function markdownIds(source) {
  const ids = new Set();
  for (const line of source.split('\n')) {
    const heading = line.match(/^ {0,3}#{1,6}\s+(.+?)(?:\s+#+\s*)?$/);
    if (!heading) continue;
    const text = decodeEntities(heading[1].trim()
      .replace(/<[^>]*>/g, '')
      .replace(/!?\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)/g, '$1')
      .replace(/\\([!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~])/g, '$1')
      .replace(/[*`]/g, ''));
    // GitHub removes punctuation before replacing spaces; "A & B" becomes
    // "a--b". Do not collapse the remaining spaces or trim emoji-only prefixes.
    const base = text.toLowerCase().replace(/[^\p{L}\p{M}\p{N}_\s-]/gu, '').replace(/\s/g, '-');
    let id = base, suffix = 0;
    while (ids.has(id)) id = `${base}-${++suffix}`;
    ids.add(id);
  }
  return new Set([...ids, ...htmlIds(source)]);
}

function htmlIds(source) {
  const ids = [...source.matchAll(/(?<![\w:-])id\s*=\s*(["'])(.*?)\1/g)].map(match => decodeEntities(match[2]));
  for (const match of source.matchAll(/<a\b[^>]*\sname\s*=\s*(["'])(.*?)\1[^>]*>/gi)) ids.push(decodeEntities(match[2]));
  return ids;
}

function references(source, markdown) {
  const refs = [];
  const add = (index, value) => refs.push({line: source.slice(0, index).split('\n').length, value: decodeEntities(value)});
  for (const match of source.matchAll(/\b(?:href|src)\s*=\s*(["'])(.*?)\1/g)) add(match.index, match[2]);
  if (markdown) {
    for (const match of source.matchAll(/!?\[[^\n]*?\]\((?:<([^>\n]*)>|((?:[^\s()]|\([^\s()]*\))+))(?:\s+["'][^\n]*?["'])?\)/g)) add(match.index, match[1] ?? match[2]);
  }
  return refs;
}

const sourceCache = new Map();
async function document(path) {
  if (!sourceCache.has(path)) {
    const source = await readFile(path, 'utf8');
    const markdown = extname(path).toLowerCase() === '.md';
    const visible = markdown ? withoutFences(source) : source;
    sourceCache.set(path, {source, visible, ids: markdown ? markdownIds(visible) : new Set(htmlIds(visible))});
  }
  return sourceCache.get(path);
}

let anchorCount = 0, resourceCount = 0, tableCount = 0, rowCount = 0;
for (const filename of ['README.md', 'index.html', ...staticPages]) {
  const path = resolve(root, filename);
  const doc = await document(path);
  const markdown = extname(path) === '.md';
  for (const {line, value} of references(doc.visible, markdown)) {
    if (!value || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(value)) continue;
    let url;
    try { url = new URL(value, 'https://local.invalid/' + filename); }
    catch { report(filename, line, `Invalid local reference: ${value}`); continue; }
    let target;
    try { target = resolve(root, '.' + decodeURIComponent(url.pathname)); }
    catch { report(filename, line, `Invalid path encoding: ${value}`); continue; }
    if (target !== root && !target.startsWith(root + sep)) { report(filename, line, `Reference escapes repository: ${value}`); continue; }
    try {
      let info = await stat(target);
      if (info.isDirectory()) { target = resolve(target, 'index.html'); info = await stat(target); }
      if (!info.isFile()) throw new Error('not a file');
      resourceCount++;
    } catch { report(filename, line, `Missing local resource: ${value}`); continue; }
    if (url.hash && ['.md', '.html'].includes(extname(target).toLowerCase())) {
      let fragment;
      try { fragment = decodeURIComponent(url.hash.slice(1)); }
      catch { report(filename, line, `Invalid fragment encoding: ${value}`); continue; }
      const targetDoc = await document(target);
      if (!targetDoc.ids.has(fragment)) report(filename, line, `Missing fragment target: ${value}`);
      anchorCount++;
    }
  }
  if (markdown) {
    const lines = doc.visible.split('\n');
    for (let index = 0; index < lines.length; index++) {
      if (!/^\s*\|/.test(lines[index])) continue;
      const start = index, rows = [];
      while (index < lines.length && /^\s*\|/.test(lines[index])) rows.push({cells: splitRow(lines[index]), line: ++index});
      index--;
      if (rows.length < 2 || !rows[1].cells.every(cell => /^:?-{3,}:?$/.test(cell))) {
        report(filename, start + 1, 'Table is missing a valid header delimiter row.');
        continue;
      }
      tableCount++;
      const width = rows[0].cells.length;
      for (const row of rows) {
        if (row.cells.length !== width) report(filename, row.line, `Table has ${row.cells.length} columns; expected ${width}.`);
        rowCount++;
      }
    }
  } else {
    const ids = htmlIds(doc.visible);
    if (new Set(ids).size !== ids.length) report(filename, 1, 'Duplicate HTML id/name attributes.');
  }
}

// These examples protect the GitHub anchor behavior that caused broken links.
assert(markdownIds('# A & B\n# A + B\n# 📢 News\n# Pixel Encoding\n# Pixel Encoding').has('a--b-1'));
assert(markdownIds('# A & B\n# 📢 News\n# Pixel Encoding\n# Pixel Encoding').has('-news'));
assert(markdownIds('# Pixel Encoding\n# Pixel Encoding').has('pixel-encoding-1'));
assert(!withoutFences('```md\n| example |\n```').includes('| example |'));

if (process.argv.includes('--built')) {
  const {files} = await build({write: false});
  const verify = async (path, expected, label) => {
    try {
      if (!Buffer.from(await readFile(path)).equals(Buffer.from(expected))) report(label, 1, 'Generated file is stale. Run npm run build.');
    } catch { report(label, 1, 'Generated file is missing. Run npm run build.'); }
  };
  for (const [name, data] of files) {
    await verify(resolve(root, name), data, name);
    await verify(resolve(root, 'dist', name), data, `dist/${name}`);
  }
  try {
    for (const entry of await readdir(resolve(root, 'dist'), {recursive: true, withFileTypes: true})) {
      if (!entry.isFile()) continue;
      const name = relative(resolve(root, 'dist'), resolve(entry.parentPath, entry.name)).split(sep).join('/');
      if (!files.has(name)) report(`dist/${name}`, 1, 'Unexpected file in generated artifact. Run npm run build.');
    }
  } catch { /* Missing output is reported by the per-file checks above. */ }
}

if (problems.length) {
  console.error(problems.join('\n'));
  process.exitCode = 1;
} else {
  console.log(`PASS: ${tableCount} tables / ${rowCount} rows, ${anchorCount} local fragments, ${resourceCount} local references${process.argv.includes('--built') ? ', and root/dist build synchronization' : ''}.`);
}
