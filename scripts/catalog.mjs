import { createHash } from 'node:crypto';

export const collections = [
  { id: 'traditional', label: 'Traditional', fullName: 'Traditional Multimodal Models', description: 'Representations, alignment, fusion, and multimodal pretraining.' },
  { id: 'mllms', label: 'MLLMs', fullName: 'Multimodal Large Language Models', description: 'Extending language models through vision adapters and multimodal interfaces.' },
  { id: 'umms', label: 'UMMs', fullName: 'Unified Multimodal Models', description: 'Understanding and generation within one unified framework.' },
  { id: 'nmms', label: 'NMMs', fullName: 'Native Multimodal Models', description: 'Architectural integration and multimodal optimization, with explicit training qualifications.' },
  { id: 'closed', label: 'Closed-source', fullName: 'Closed-Source Multimodal Models', description: 'Proprietary models, organized by the release years recorded in the list.' },
];
export const repo = 'https://github.com/OpenEnvision/Awesome-Multimodal-Modeling';
export const escapeHTML = (value = '') => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function plain(value = '') {
  return value.replace(/!\[[^\]]*\]\([^)]*\)/g, '').replace(/\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)/g, '$1')
    .replace(/<br\s*\/?\s*>/gi, ' · ').replace(/<[^>]*>/g, '').replace(/[*`]/g, '').replace(/\\([|_&])/g, '$1')
    .replace(/&amp;/g, '&').replace(/&nbsp;/g, ' ').replace(/\s+/g, ' ').trim();
}
export function splitRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split(/(?<!\\)\|/).map(c => c.trim().replace(/\\\|/g, '|'));
}
export function linksFrom(value) {
  const links = [];
  for (const m of value.matchAll(/(?<!!)\[([^\]]+)\]\((https?:\/\/(?:[^\s()]|\([^)]*\))+)\)/g)) {
    try { const url = new URL(m[2]); if (['http:', 'https:'].includes(url.protocol)) links.push({label: plain(m[1]), url: url.href}); } catch { /* Invalid source links remain visible in source metadata. */ }
  }
  return links.filter((a, i) => links.findIndex(b => b.url === a.url) === i);
}
const cleanHeading = text => plain(text).replace(/^\d+(?:\.\d+)*\.?\s*/, '');
const slug = text => text.toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-');

export function parseCatalog(markdown) {
  const lines = markdown.split(/\r?\n/), entries = [], sections = new Map(), usedIds = new Set();
  let stack = [], active = null, headers = null, fence = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*(```|~~~)/.test(line)) { fence = !fence; headers = null; continue; }
    if (fence) continue;
    const heading = line.match(/^(#{2,5})\s+(.+?)\s*$/);
    if (heading) {
      const level = heading[1].length;
      headers = null;
      if (level === 2) {
        const number = Number(heading[2].match(/^(\d+)\./)?.[1]);
        active = collections[number - 2] ?? null;
        stack = [];
      } else if (active) {
        stack = stack.filter(h => h.level < level);
        const label = cleanHeading(heading[2]);
        const id = [active.id, ...stack.map(s => slug(s.label)), slug(label)].join('/');
        const section = {id, label, level, collection: active.id, parent: stack.at(-1)?.id ?? active.id};
        stack.push(section);
        sections.set(id, section);
      }
      continue;
    }
    if (!active || !line.trim().startsWith('|')) { headers = null; continue; }
    const cells = splitRow(line);
    if (cells.every(c => /^:?-+:?$/.test(c))) continue;
    if (['Paper', 'Model', 'Title'].includes(cells[0]) && cells.includes('Links')) { headers = cells; continue; }
    if (!headers) continue;
    let rowHeaders = [...headers];
    if (cells.length === rowHeaders.length + 1 && rowHeaders[3] === 'Notes' && /^\d[\d.,\s–-]*[BMTK]\b/.test(cells[3])) rowHeaders.splice(3, 0, 'Scale');
    const fields = cells.map((c, index) => ({label: rowHeaders[index] ?? `Additional notes ${index - rowHeaders.length + 1}`, value: plain(c)}));
    const get = name => fields.find(f => f.label === name)?.value ?? '';
    const title = plain(cells[0]);
    const venue = get('Venue') || (rowHeaders[0] === 'Model' ? get('Paper') : '');
    const year = Number(venue.match(/\b(?:19|20)\d{2}\b/)?.[0] || stack.map(s => s.label).join(' ').match(/\b20\d{2}\b/)?.[0]) || null;
    const date = venue.match(/\b20\d{2}-\d{2}-\d{2}\b/)?.[0] ?? (year ? `${year}-00-00` : '0000-00-00');
    const resources = linksFrom(cells[rowHeaders.indexOf('Links')] ?? '');
    const section = stack.at(-1)?.id ?? active.id;
    let id = createHash('sha256').update(`${section}|${title}|${venue}|${resources[0]?.url ?? ''}`).digest('hex').slice(0, 12);
    while (usedIds.has(id)) id += 'x';
    usedIds.add(id);
    entries.push({id, title, venue, year, date, collection: active.id, section, path: stack.map(s => s.label), ancestors: stack.map(s => s.id),
      notes: get('Notes') || get('Insights') || get('Focus'), tasks: get('Task'), resources, fields,
      kind: stack.some(s => s.label === 'Design Analyses & Scaling Laws') ? 'Analysis' : 'Research entry',
      sourceLine: i + 1, sourceUrl: `${repo}/blob/main/README.md?plain=1#L${i + 1}`});
  }
  if (!entries.length) throw new Error('No catalog entries found; check README table structure.');
  return {
    collections: collections.map(c => ({...c, count: entries.filter(e => e.collection === c.id).length})),
    sections: [...sections.values()].map(s => ({...s, count: entries.filter(e => e.ancestors.includes(s.id)).length})).filter(s => s.count),
    entries,
    years: [...new Set(entries.map(e => e.year).filter(Boolean))].sort((a,b) => b-a),
    sourceHash: createHash('sha256').update(markdown).digest('hex'),
  };
}

if(process.argv.includes('--check-catalog')){
  const {readFileSync}=await import('node:fs');
  const {default:assert}=await import('node:assert/strict');
  const {readState,selectEntries}=await import('../assets/site/library.js');
  const md=readFileSync(new URL('../README.md',import.meta.url),'utf8'),c=parseCatalog(md),lines=md.split('\n');
  const a=lines.findIndex(l=>l.startsWith('## 2. ')),b=lines.findIndex(l=>l.startsWith('## 7. '));
  const expected=lines.map((l,i)=>({l,i:i+1})).slice(a,b).filter(({l})=>l.startsWith('|')&&!/^\|\s*(?:-+|Paper\s*\||Model\s*\||Title\s*\|)/.test(l)).map(x=>x.i);
  assert.deepEqual(c.entries.map(e=>e.sourceLine),expected);
  assert.equal(new Set(c.entries.map(e=>e.id)).size,c.entries.length);
  assert.equal(c.collections.reduce((n,x)=>n+x.count,0),c.entries.length);
  for(const s of c.sections) assert.equal(selectEntries(c,readState('?section='+encodeURIComponent(s.id),c)).length,s.count);
  assert.deepEqual(splitRow('| A \\| B | C |'),['A | B','C']);
  assert.equal(linksFrom('[a](https://example.org/a_(b)) [bad](javascript:alert(1))').length,1);
  assert.equal(escapeHTML('<img>'),'&lt;img&gt;');
  assert(c.entries.every(e=>e.resources.length&&e.title));
  assert(c.entries.filter(e=>e.fields.some(f=>f.label==='Focus')).every(e=>e.notes));
  const t=c.entries.find(e=>e.title.startsWith('Transfusion:')&&e.collection==='umms');
  if(t){assert(t.fields.some(f=>f.label==='Scale'&&f.value==='7B-scale report'));assert(t.tasks.includes('generation'));}
  const safe=readState('?page=-1&sort=bad&collection=bad',c);
  assert.equal(safe.page,1);assert.equal(safe.sort,'readme');assert.equal(safe.collection,'');
  console.log('PASS: '+c.entries.length+' source rows, '+c.sections.length+' category filters, IDs, metadata, legacy table formats, URL validation, and HTML escaping.');
}
