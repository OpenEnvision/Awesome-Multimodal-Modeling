import {PAGE_SIZE, esc, normalize, selectEntries, readState, stateSearch, rowHTML, resourceHTML, bookmark} from './library.js';

const $ = selector => document.querySelector(selector);
const dialog = $('#entry-dialog');
let catalog, state, saved = new Set(), activeEntry = '', opener = null, toastTimer, searchTimer;
const expanded = new Set();
function toast(message) {
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 3500);
}
function syncURL(push = false, hash = location.hash) {
  const next = location.pathname + stateSearch(state, activeEntry) + hash;
  if (next !== location.pathname + location.search + location.hash) history[push ? 'pushState' : 'replaceState']({}, '', next);
}
function sidebarChildren(parent) {
  return catalog.sections.filter(s => s.parent === parent).map(s => {
    const children = sidebarChildren(s.id);
    return `<div><button class="category-button sub-button ${state.section === s.id ? 'active' : ''}" data-section="${esc(s.id)}" aria-pressed="${state.section === s.id}"><span>${esc(s.label)}</span><span>${s.count}</span></button>${children ? `<div class="subcategories">${children}</div>` : ''}</div>`;
  }).join('');
}
function renderSidebar() {
  $('#category-tree').innerHTML = `<h3 class="sidebar-heading">Collections</h3><button class="category-button ${!state.collection && !state.saved ? 'active' : ''}" data-collection="" aria-pressed="${!state.collection && !state.saved}"><span>All entries</span><span>${catalog.entries.length}</span></button>` + catalog.collections.map(c => {
    const open = expanded.has(c.id) || state.collection === c.id;
    return `<div class="category-row"><button class="category-button ${state.collection === c.id && !state.section ? 'active' : ''}" data-collection="${c.id}" aria-pressed="${state.collection === c.id && !state.section}"><span>${c.label}</span><span>${c.count}</span></button><button class="expand-category" data-expand="${c.id}" aria-label="${open ? 'Collapse' : 'Expand'} ${c.label} subcategories" aria-expanded="${open}" aria-controls="sub-${c.id}">›</button></div><div class="subcategories" id="sub-${c.id}" ${open ? '' : 'hidden'}>${sidebarChildren(c.id)}</div>`;
  }).join('') + `<div class="saved-filter"><button class="category-button ${state.saved ? 'active' : ''}" data-saved-view aria-pressed="${state.saved}"><span>Saved entries</span>${bookmark}<span>${saved.size}</span></button></div>`;
}
function renderChips() {
  const chips = [];
  if (state.q) chips.push(['q', `Search: ${state.q}`]);
  if (state.section) chips.push(['section',catalog.sections.find(s=>s.id===state.section).label]);
  if (state.year) chips.push(['year',state.year]);
  if (state.resource) chips.push(['resource', {paper:'Paper links',code:'Code links',hf:'Hugging Face links'}[state.resource]]);
  const box = $('#active-filters');
  box.hidden = !chips.length;
  box.innerHTML = chips.map(([key,label])=>`<button class="filter-chip" data-clear="${key}" aria-label="Remove filter ${esc(label)}">${esc(label)}<span aria-hidden="true">×</span></button>`).join('') + (chips.length ? '<button class="clear-all" data-reset>Reset all filters</button>' : '');
}
function render({sidebar = true} = {}) {
  const entries = selectEntries(catalog, state, saved);
  const pages = Math.max(1, Math.ceil(entries.length / PAGE_SIZE));
  state.page = Math.min(state.page, pages);
  const start = (state.page - 1) * PAGE_SIZE;
  const section = catalog.sections.find(s => s.id === state.section);
  const collection = catalog.collections.find(c => c.id === state.collection);
  $('#search').value = state.q;
  $('#year').value = state.year;
  $('#sort').value = state.sort;
  $('#resource').value = state.resource;
  $('#results-title').textContent = state.saved ? 'Saved entries' : section?.label || collection?.label || 'All entries';
  $('#results-description').textContent = state.saved ? 'Your reading list, saved in this browser.' : section ? section.id.split('/').length > 2 ? `${collection.fullName} · ${catalog.sections.find(s=>s.id===section.parent)?.label || ''}` : collection.fullName : collection?.description || 'Browse the complete collection.';
  $('#result-count').textContent = `${entries.length} ${entries.length === 1 ? 'entry' : 'entries'}${state.q ? ` matching “${state.q}”` : ''}`;
  $('#entries').innerHTML = entries.length ? entries.slice(start,start+PAGE_SIZE).map(e=>rowHTML(e,saved.has(e.id))).join('') : `<div class="empty-state"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="7"/><path d="m16 16 5 5"/></svg><h4>${state.saved && !saved.size ? 'A place for your next read.' : 'No entries found.'}</h4><p>${state.saved && !saved.size ? 'Use the bookmark beside an entry to keep it here.' : 'Try a different search, year, or collection.'}</p><button class="button" data-reset>Browse all entries</button></div>`;
  let pagination = '';
  if (entries.length) {
    const indices = new Set([1,pages,state.page-1,state.page,state.page+1].filter(p=>p>=1&&p<=pages));
    let last = 0;
    const buttons = [...indices].sort((a,b)=>a-b).map(p=> {
      const gap = p-last>1 ? '<span class="page-gap" aria-hidden="true">…</span>' : '';
      last = p;
      return `${gap}<button class="page-button" data-page="${p}" aria-label="Page ${p}" ${p===state.page ? 'aria-current="page"' : ''}>${p}</button>`;
    }).join('');
    pagination = `<span>Showing ${start+1}–${Math.min(start+PAGE_SIZE,entries.length)} of ${entries.length} entries</span><nav class="page-buttons" aria-label="Results pages"><button class="page-button" data-page="${state.page-1}" ${state.page===1 ? 'disabled' : ''} aria-label="Previous page">← Prev</button>${buttons}<button class="page-button" data-page="${state.page+1}" ${state.page===pages ? 'disabled' : ''} aria-label="Next page">Next →</button></nav>`;
  }
  $('#pagination').innerHTML = pagination;
  renderChips();
  if (sidebar) renderSidebar();
  syncURL();
}
function update(patch, {push = true, focus = false, hash = location.hash} = {}) {
  clearTimeout(searchTimer);
  if (patch.q === undefined && document.activeElement === $('#search')) patch.q = $('#search').value;
  state = {...state, ...patch, page: patch.page ?? 1};
  activeEntry = '';
  syncURL(push, hash);
  render();
  if (focus) { $('#results-title').focus({preventScroll:true}); $('#results-title').scrollIntoView({block:'start',behavior:'auto'}); }
}
function detail(e, trigger = null, push = true) {
  if (!e) return;
  if (trigger) opener = trigger;
  activeEntry = e.id;
  syncURL(push);
  const collection = catalog.collections.find(c=>c.id===e.collection);
  // Cross-references are conservative: same leading model name and the same paper URL.
  const paperKey = entry => entry.resources.find(r=>/arxiv\.org\/(abs|pdf)\//.test(r.url))?.url.replace('/pdf/','/abs/').replace(/\.pdf$/,'').replace(/v\d+$/,'');
  const related = catalog.entries.filter(other => other.id !== e.id && normalize(other.title.split(':')[0]) === normalize(e.title.split(':')[0]) && paperKey(e) && paperKey(e) === paperKey(other));
  $('#detail-content').innerHTML = `<p class="detail-category">${esc(collection.label)} / ${esc(e.path.join(' / '))}</p><h2 id="detail-title">${esc(e.title)}</h2><div class="detail-resources">${e.resources.map(resourceHTML).join('')}</div><dl class="detail-fields">${e.fields.filter((f,i)=>i>0 && f.label !== 'Links' && f.value && f.value !== '—' && f.value !== '-').map(f=>`<dt>${esc(f.label)}</dt><dd>${esc(f.value)}</dd>`).join('')}</dl>${related.length ? `<div class="related-entries"><p>Also listed in the source</p>${related.map(r=>`<button data-detail="${r.id}">${esc(catalog.collections.find(c=>c.id===r.collection).label)} / ${esc(r.path.at(-1))} ↗</button>`).join('')}</div>` : ''}<div class="detail-actions"><button class="button" data-detail-save="${e.id}" aria-pressed="${saved.has(e.id)}">${saved.has(e.id) ? 'Saved to your reading list' : 'Save to reading list'} ${bookmark}</button><a href="${esc(e.sourceUrl)}" target="_blank" rel="noopener noreferrer">View original README entry ↗</a></div>`;
  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0;
}
function closeDetail() {
  activeEntry = '';
  if (dialog.open) dialog.close();
  syncURL();
  const restored = opener?.isConnected ? opener : document.querySelector(`[data-detail="${opener?.dataset.detail || ''}"]`);
  restored?.focus({preventScroll:true});
}
function save(id) {
  const focusedSave = document.activeElement?.getAttribute('data-save');
  if (saved.has(id)) saved.delete(id); else saved.add(id);
  let persisted = true;
  try { localStorage.setItem('openenvision-reading-list-v1', JSON.stringify([...saved])); } catch { persisted = false; }
  render();
  if (activeEntry) { detail(catalog.entries.find(e=>e.id===activeEntry),null,false); $('[data-detail-save]')?.focus({preventScroll:true}); }
  else if (focusedSave) document.querySelector(`[data-save="${focusedSave}"]`)?.focus({preventScroll:true});
  toast(persisted ? saved.has(id) ? 'Saved to your reading list.' : 'Removed from your reading list.' : 'Saved for this visit. Browser storage is unavailable.');
}
async function init() {
  try {
    const response = await fetch(new URL('./catalog.json', import.meta.url));
    if (!response.ok) throw new Error(`Catalog request failed: ${response.status}`);
    catalog = await response.json();
    state = readState(location.search, catalog);
    try {
      const stored = JSON.parse(localStorage.getItem('openenvision-reading-list-v1') || '[]');
      saved = new Set((Array.isArray(stored) ? stored : []).filter(id=>catalog.entries.some(e=>e.id===id)));
    } catch { saved = new Set(); }
    const initialEntry = new URLSearchParams(location.search).get('entry');
    // Keep a deep link while the first render synchronizes filter parameters.
    activeEntry = catalog.entries.some(e=>e.id===initialEntry) ? initialEntry : '';
    render();
    document.querySelectorAll('#search-form :disabled, #resource, #share-view').forEach(el=>el.disabled=false);
    if (activeEntry) detail(catalog.entries.find(e=>e.id===activeEntry),null,false);
    $('#search-form').addEventListener('submit', event=>{event.preventDefault();clearTimeout(searchTimer);update({q:$('#search').value},{push:false});});
    $('#search').addEventListener('input', event=>{const value=event.target.value;clearTimeout(searchTimer);searchTimer=setTimeout(()=>update({q:value},{push:false}),120);});
    for (const key of ['year','sort','resource']) $(`#${key}`).addEventListener('change', event=>update({[key]:event.target.value}));
    document.addEventListener('click', event=>{
      const target = event.target.closest('button, a');
      if (!target || target.disabled) return;
      if (target.hasAttribute('data-collection')) {
        if(target.tagName==='A'&&(event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)) return;
        const fromTaxonomy=Boolean(target.closest('#taxonomy'));
        event.preventDefault();update({collection:target.dataset.collection,section:'',saved:false},{hash:fromTaxonomy?'#library':location.hash});
        $('#category-sidebar').classList.remove('mobile-open');$('#mobile-categories').setAttribute('aria-expanded','false');
        if (fromTaxonomy) { $('#results-title').focus({preventScroll:true}); $('#library').scrollIntoView({behavior:'auto'}); }
      } else if (target.hasAttribute('data-section')) {const section=catalog.sections.find(s=>s.id===target.dataset.section); update({collection:section.collection,section:section.id,saved:false});}
      else if (target.hasAttribute('data-expand')) {
        const id=target.dataset.expand, el=$(`#sub-${id}`), open=el.hidden;
        el.hidden=!open;target.setAttribute('aria-expanded',String(open));target.setAttribute('aria-label',`${open?'Collapse':'Expand'} ${catalog.collections.find(c=>c.id===id).label} subcategories`);
        if(open)expanded.add(id);else expanded.delete(id);
      } else if (target.hasAttribute('data-saved-view')) update({saved:true,collection:'',section:'',q:'',year:'',resource:''});
      else if (target.hasAttribute('data-save')) save(target.dataset.save);
      else if (target.hasAttribute('data-detail-save')) save(target.dataset.detailSave);
      else if (target.hasAttribute('data-detail')) detail(catalog.entries.find(e=>e.id===target.dataset.detail),target);
      else if (target.hasAttribute('data-page')) update({page:Number(target.dataset.page)},{focus:true});
      else if (target.hasAttribute('data-clear')) update({[target.dataset.clear]:''});
      else if (target.hasAttribute('data-reset')) update({q:'',collection:'',section:'',year:'',resource:'',saved:false,sort:'readme'});
    });
    $('#mobile-categories').addEventListener('click',()=>{const open=$('#category-sidebar').classList.toggle('mobile-open');$('#mobile-categories').setAttribute('aria-expanded',String(open));});
    $('#share-view').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(location.href);toast('Link copied. Your filters are included.');}catch{toast('Copy the page address to share this view.');}});
    $('#close-dialog').addEventListener('click',closeDetail);
    dialog.addEventListener('cancel',event=>{event.preventDefault();closeDetail();});
    dialog.addEventListener('click',event=>{if(event.target===dialog){const box=dialog.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)closeDetail();}});
    document.addEventListener('keydown',event=>{if(event.key==='/'&&!dialog.open&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&!['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)&&!document.activeElement.isContentEditable){event.preventDefault();$('#search').focus();}});
    window.addEventListener('popstate',()=>{clearTimeout(searchTimer);state=readState(location.search,catalog);const id=new URLSearchParams(location.search).get('entry');activeEntry=catalog.entries.some(e=>e.id===id)?id:'';render();if(activeEntry)detail(catalog.entries.find(e=>e.id===activeEntry),null,false);else if(dialog.open)dialog.close();});
  } catch (error) { $('#load-error').hidden=false; console.error('Research library:',error); }
}
init();
