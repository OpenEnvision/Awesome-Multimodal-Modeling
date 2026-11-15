import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {parseCatalog} from './catalog.mjs';
import {migrateSavedEntries, readState, resolveEntryId, resourceKind, rowHTML, selectEntries, stateSearch} from '../assets/site/library.js';

const noMigrations = {identities: {entryAliases: {}, sectionAliases: {}}};
const source = ({collection = '2. Traditional', section = 'Original category', title = 'Original model', venue = 'arXiv 2020', url = 'https://example.org/original', id = 'persistent-123', label = 'Paper', notes = 'A model'} = {}) => `
## ${collection}
### ${section}
| Paper | Venue | Links | Notes | Task |
| --- | --- | --- | --- | --- |
| ${title} <!-- entry:${id} --> | ${venue} | [${label}](${url}) | ${notes} | Understanding |
`;

test('an existing bookmark and detail URL survive simultaneous title, venue, primary URL and collection changes', () => {
  const before = parseCatalog(source(), noMigrations);
  const after = parseCatalog(source({collection: '3. MLLMs', section: 'Replacement category', title: 'Corrected title', venue: 'ICLR 2025', url: 'https://example.org/corrected'}), noMigrations);
  const bookmarked = JSON.parse(JSON.stringify([before.entries[0].id]));
  const sharedURL = stateSearch(readState('', before), before.entries[0].id);
  const afterState = readState('?saved=1', after);
  const saved = migrateSavedEntries(after, bookmarked);
  assert.deepEqual(selectEntries(after, afterState, saved).map(e => e.title), ['Corrected title']);
  const detailId = resolveEntryId(after, new URLSearchParams(sharedURL).get('entry'));
  assert.equal(after.entries.find(e => e.id === detailId)?.title, 'Corrected title');
  assert.equal(after.entries[0].collection, 'mllms');
  assert.equal(after.entries[0].id, before.entries[0].id);
  assert(after.entries[0].fields.every(f => !f.value.includes('<!--')));
});

test('entries sharing a paper and title keep separate identities, while missing or duplicated identities fail explicitly', () => {
  const row = '| Original model <!-- entry:second-entry --> | arXiv 2020 | [Paper](https://example.org/original) | A distinct listing | Understanding |';
  const catalog = parseCatalog(source() + row, noMigrations);
  assert.equal(catalog.entries.length, 2);
  assert.notEqual(catalog.entries[0].id, catalog.entries[1].id);
  assert.throws(() => parseCatalog(source().replace('<!-- entry:persistent-123 -->', ''), noMigrations), /needs one immutable/);
  assert.throws(() => parseCatalog(source() + row.replace('second-entry', 'persistent-123'), noMigrations), /duplicate entry ID/);
  assert.throws(() => parseCatalog(source({id: 'invalid_id'}), noMigrations), /needs one immutable/);
});

test('retired IDs resolve only through an explicit migration, including cross-collection corrections', () => {
  const catalog = parseCatalog(source(), {identities: {entryAliases: {'retired-entry': 'persistent-123'}}});
  assert.equal(resolveEntryId(catalog, 'retired-entry'), 'persistent-123');
  assert.equal(resolveEntryId(catalog, 'missing-entry'), '');
  assert.equal(resolveEntryId(catalog, null), '');
  assert.deepEqual([...migrateSavedEntries(catalog, ['retired-entry', 'persistent-123', 'missing-entry', null])], ['persistent-123']);
  assert.deepEqual([...migrateSavedEntries(catalog, {id: 'persistent-123'})], []);
  assert.throws(() => parseCatalog(source(), {identities: {entryAliases: {'retired-entry': 'missing-entry'}}}), /Invalid entry migration/);
});

test('renamed category links keep their filter and reject an incompatible collection', () => {
  const oldSection = 'traditional/previous-category';
  const section = 'traditional/original-category';
  const catalog = parseCatalog(source(), {identities: {sectionAliases: {[oldSection]: section}}});
  const migrated = readState('?section=' + encodeURIComponent(oldSection), catalog);
  assert.equal(migrated.section, section);
  assert.equal(selectEntries(catalog, migrated).length, 1);
  assert.equal(readState('?collection=mllms&section=' + encodeURIComponent(oldSection), catalog).section, '');
});

test('retired production entries retain their original deep links and bookmarks', () => {
  const catalog = parseCatalog(readFileSync(new URL('../README.md', import.meta.url), 'utf8'));
  const expected = [
    ['804e9375cfd3', '9fac946b2b27', 'traditional'],
    ['a93adb5a3e62', '9fac946b2b27', 'traditional'],
    ['521128b79b82', '64f43671fa13', 'traditional'],
    ['ab98a6468b1e', '900d8adb0e86', 'mllms'],
    ['1317e9b32a52', '2f5bd280c1ba', 'mllms'],
    ['0eb7c6ae7c9a', '8403ff00096e', 'mllms'],
  ];
  for (const [oldId, id, collection] of expected) {
    assert.equal(resolveEntryId(catalog, oldId), id);
    const saved = migrateSavedEntries(catalog, [oldId]);
    const result = selectEntries(catalog, readState('?saved=1', catalog), saved);
    assert.equal(result.length, 1);
    assert.equal(result[0].id, id);
    assert.equal(result[0].collection, collection);
  }
  for (const [oldSection, section] of Object.entries(catalog.sectionAliases)) {
    const state = readState('?section=' + encodeURIComponent(oldSection), catalog);
    assert.equal(state.section, section);
    assert(selectEntries(catalog, state).length > 0);
    assert.equal(new URLSearchParams(stateSearch(state)).get('section'), section);
  }
});

test('code filtering recognizes GitHub destinations and HF detection uses the hostname', () => {
  assert.equal(resourceKind({label: 'HF / Official', url: 'https://github.com/PaddlePaddle/PaddleOCR'}), 'code');
  assert.equal(resourceKind({label: 'Implementation', url: 'https://example.org/repository'}), 'code');
  assert.equal(resourceKind({label: 'HF', url: 'https://huggingface.co/org/model'}), 'hf');
  assert.equal(resourceKind({label: 'HF', url: 'https://huggingface.co.evil.example/model'}), 'other');
  assert.equal(resourceKind({label: 'HF', url: 'https://example.org/huggingface.co/model'}), 'other');
  assert.equal(resourceKind({label: 'Official', url: 'https://github.com.evil.example/repo'}), 'other');
  assert.equal(resourceKind({label: 'Paper', url: 'https://arxiv.org/abs/2501.00001'}), 'paper');
  const catalog = parseCatalog(source({label: 'Official', url: 'https://github.com/PaddlePaddle/PaddleOCR'}), noMigrations);
  assert.equal(selectEntries(catalog, readState('?resource=code', catalog)).length, 1);
});

test('paper filtering recognizes reports and proceedings across hosts', () => {
  const report = source({id: 'hosted-report', label: 'Report', url: 'https://huggingface.co/org/model/blob/main/technical-report.pdf'});
  const proceedings = source({id: 'conference-paper', label: 'Proceedings', url: 'https://aclanthology.org/2026.acl-long.1/'});
  const code = source({id: 'hosted-code', label: 'Implementation', url: 'https://huggingface.co/org/model/blob/main/modeling.py'});
  const status = source({id: 'release-status', label: 'Release status', url: 'https://github.com/org/repo'});
  const catalog = parseCatalog(report + proceedings + code + status, noMigrations);
  const idsFor = kind => selectEntries(catalog, readState(`?resource=${kind}`, catalog)).map(e => e.id);
  assert.deepEqual(idsFor('paper'), ['hosted-report', 'conference-paper']);
  assert.deepEqual(idsFor('hf'), ['hosted-report', 'hosted-code']);
  assert.deepEqual(idsFor('code'), ['hosted-code']);
  for (const label of ['Announcement', 'Release status', 'Release history', 'Project', 'Model card']) {
    assert.equal(resourceKind({label, url: 'https://github.com/org/repo'}), 'other');
  }
});

test('only the dedicated Resources evaluation section enters the website catalog', () => {
  const markdown = `${source()}
## 7. Resources
### 7.2 Slides & Survey Papers
| Paper | Venue | Links | Notes | Task |
| --- | --- | --- | --- | --- |
| Uncatalogued survey | arXiv 2025 | [Paper](https://example.org/survey) | A survey | Survey |
### 7.4 Benchmarks & Evaluation
#### Interleaved Evaluation
| Paper | Venue | Links | Notes | Task |
| --- | --- | --- | --- | --- |
| A benchmark <!-- entry:benchmark-123 --> | ICLR 2026 | [Paper](https://example.org/benchmark) | First public: 2025-04-04; joint evaluation | Benchmark |
### 7.5 Other Resources
| Paper | Venue | Links | Notes | Task |
| --- | --- | --- | --- | --- |
| Other resource | 2026 | [Paper](https://example.org/other) | Not a catalog | Tool |
`;
  const catalog = parseCatalog(markdown, noMigrations);
  assert.equal(catalog.entries.length, 2);
  const entry = catalog.entries.find(e => e.id === 'benchmark-123');
  assert.equal(entry.collection, 'evaluation');
  assert.equal(entry.kind, 'Benchmark');
  assert.equal(entry.date, '2025-04-04');
  assert.equal(entry.year, 2026); // Year filters retain the published venue year.
  assert.equal(entry.ancestors.length, 2);
  const state = readState('?section=' + encodeURIComponent(entry.section), catalog);
  assert.equal(state.collection, 'evaluation');
  assert.deepEqual(selectEntries(catalog, state).map(e => e.id), ['benchmark-123']);
  assert(rowHTML(entry).includes('>Benchmark</span>'));
});

test('first-public dates sort independently of conference years and future release promises', () => {
  const catalog = parseCatalog(source({venue: 'NeurIPS 2026', notes: 'First public: 2025-09-01; weights expected 2026-10-31'}), noMigrations);
  assert.equal(catalog.entries[0].date, '2025-09-01');
  const release = parseCatalog(source({venue: 'Model release 2026-10-15'}), noMigrations);
  assert.equal(release.entries[0].date, '2026-10-15');
  assert.equal(parseCatalog(source(), noMigrations).entries[0].date, '2020-00-00');
});

test('month-only dates keep release order without inventing a release day', () => {
  const releases = [
    source({id: 'september-release', venue: 'Model release 2026-09-01'}),
    source({id: 'august-month', venue: 'Model release 2026-08, report 2026-11-01'}),
    source({id: 'july-release', venue: 'Model release 2026-07-31'}),
    source({id: 'august-release', venue: 'Model release 2026-08-01'}),
    source({id: 'first-public-month', venue: 'NeurIPS 2026, arXiv 2026-08-15', notes: 'First public: 2025-12 (code)'}),
  ];
  const catalog = parseCatalog(releases.join(''), noMigrations);
  const oldest = selectEntries(catalog, readState('?sort=oldest', catalog));
  const newest = selectEntries(catalog, readState('?sort=newest', catalog));
  assert.deepEqual(oldest.map(e => [e.id, e.date]), [
    ['first-public-month', '2025-12'],
    ['july-release', '2026-07-31'],
    ['august-month', '2026-08'],
    ['august-release', '2026-08-01'],
    ['september-release', '2026-09-01'],
  ]);
  assert.deepEqual(newest.map(e => e.id), oldest.map(e => e.id).reverse());
  assert.equal(oldest[0].year, 2026);
});
