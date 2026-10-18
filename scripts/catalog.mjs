import { createHash } from 'node:crypto';
import identityMigrations from '../data/entry-identities.json' with { type: 'json' };

export const collections = [
  { id: 'traditional', label: 'Traditional', fullName: 'Traditional Multimodal Models', description: 'Representations, alignment, fusion, and multimodal pretraining.' },
  { id: 'mllms', label: 'MLLMs', fullName: 'Multimodal Large Language Models', description: 'Extending language models through vision adapters and multimodal interfaces.' },
  { id: 'umms', label: 'UMMs', fullName: 'Unified Multimodal Models', description: 'Understanding and generation within one unified framework.' },
  { id: 'nmms', label: 'NMMs', fullName: 'Native Multimodal Models', description: 'Architectural integration and multimodal optimization, with explicit training qualifications.' },
  { id: 'closed', label: 'Closed-source', fullName: 'Closed-Source Multimodal Models', description: 'Proprietary models, organized by the release years recorded in the list.' },
  { id: 'evaluation', label: 'Evaluation', fullName: 'Benchmarks & Evaluation', description: 'Benchmarks for understanding, generation, interleaved interaction, and multimodal robustness.' },
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

export function parseCatalog(markdown, {identities = identityMigrations} = {}) {
  const lines = markdown.split(/\r?\n/), entries = [], sections = new Map(), usedIds = new Set();
  let stack = [], active = null, chapter = null, headers = null, fence = false;
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
        chapter = number;
        active = number >= 2 && number <= 6 ? collections[number - 2] : null;
        stack = [];
      } else {
        // Only the dedicated evaluation section is a catalog within Resources.
        // Surveys and tool tables retain their separate README formats.
        if (chapter === 7 && level === 3) {
          active = cleanHeading(heading[2]) === 'Benchmarks & Evaluation' ? collections.find(c => c.id === 'evaluation') : null;
          stack = [];
        }
        if (!active) continue;
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
    const notes = get('Notes') || get('Insights') || get('Focus');
    // A conference year and a later weights release are not a paper's first date.
    const firstPublicDate = notes.match(/\bFirst public:\s*(20\d{2}-\d{2}-\d{2})\b/i)?.[1] ?? null;
    const date = firstPublicDate ?? venue.match(/\b20\d{2}-\d{2}-\d{2}\b/)?.[0] ?? (year ? `${year}-00-00` : '0000-00-00');
    const resources = linksFrom(cells[rowHeaders.indexOf('Links')] ?? '');
    const section = stack.at(-1)?.id ?? active.id;
    // An explicit source identity survives corrections to every piece of visible metadata.
    // Never guess from a title or URL: distinct entries may intentionally share either one.
    const markers = [...cells[0].matchAll(/<!--\s*entry:([^\s]+)\s*-->/g)];
    const id = markers[0]?.[1];
    if (markers.length !== 1 || !/^[a-z0-9][a-z0-9-]{5,63}$/.test(id ?? '')) {
      throw new Error(`README line ${i + 1}: each catalog row needs one immutable <!-- entry:ID --> in its first cell. Keep existing IDs when editing or moving rows; generate a new ID with node -e "console.log(require('node:crypto').randomUUID())".`);
    }
    if (usedIds.has(id)) throw new Error(`README line ${i + 1}: duplicate entry ID "${id}"; distinct rows need distinct immutable IDs.`);
    usedIds.add(id);
    entries.push({id, title, venue, year, date, collection: active.id, section, path: stack.map(s => s.label), ancestors: stack.map(s => s.id),
      notes, tasks: get('Task'), resources, fields,
      kind: active.id === 'evaluation' ? get('Type') || 'Benchmark' : stack.some(s => s.label === 'Design Analyses & Scaling Laws') ? 'Analysis' : 'Research entry',
      sourceLine: i + 1, sourceUrl: `${repo}/blob/main/README.md?plain=1#L${i + 1}`});
  }
  if (!entries.length) throw new Error('No catalog entries found; check README table structure.');
  const entryAliases = identities.entryAliases ?? {}, sectionAliases = identities.sectionAliases ?? {};
  for (const [oldId, id] of Object.entries(entryAliases)) {
    if (usedIds.has(oldId) || !usedIds.has(id)) throw new Error(`Invalid entry migration ${oldId} → ${id}: old IDs must be retired and targets must exist.`);
  }
  for (const [oldId, id] of Object.entries(sectionAliases)) {
    if (sections.has(oldId) || !sections.has(id)) throw new Error(`Invalid section migration ${oldId} → ${id}: old sections must be retired and targets must exist.`);
  }
  return {
    collections: collections.map(c => ({...c, count: entries.filter(e => e.collection === c.id).length})),
    sections: [...sections.values()].map(s => ({...s, count: entries.filter(e => e.ancestors.includes(s.id)).length})).filter(s => s.count),
    entries,
    entryAliases,
    sectionAliases,
    years: [...new Set(entries.map(e => e.year).filter(Boolean))].sort((a,b) => b-a),
    sourceHash: createHash('sha256').update(markdown).digest('hex'),
  };
}

if(process.argv.includes('--check-catalog')){
  const {readFileSync}=await import('node:fs');
  const {default:assert}=await import('node:assert/strict');
  const {readState,selectEntries}=await import('../assets/site/library.js');
  const md=readFileSync(new URL('../README.md',import.meta.url),'utf8'),c=parseCatalog(md),lines=md.split('\n');
  const expected=[];
  let exampleFence=false;
  for (const [i,line] of lines.entries()) {
    if (/^\s*(```|~~~)/.test(line)) { exampleFence=!exampleFence; continue; }
    if (!exampleFence && line.startsWith('|') && /<!--\s*entry:/.test(line)) expected.push(i+1);
  }
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
  if(t){assert(t.notes.includes('7B-scale report'));assert(t.tasks.includes('generation'));assert(!t.fields.some(f=>f.label.startsWith('Additional notes')));}
  const safe=readState('?page=-1&sort=bad&collection=bad',c);
  assert.equal(safe.page,1);assert.equal(safe.sort,'readme');assert.equal(safe.collection,'');
  console.log('PASS: '+c.entries.length+' source rows, '+c.sections.length+' category filters, IDs, metadata, legacy table formats, URL validation, and HTML escaping.');
}
