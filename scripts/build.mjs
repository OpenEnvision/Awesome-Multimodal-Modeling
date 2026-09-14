import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {dirname, resolve} from 'node:path';
import {parseCatalog, escapeHTML as esc} from './catalog.mjs';
import {rowHTML} from '../website/library.js';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export async function build({write=true}={}) {
  const markdown = await readFile(resolve(root,'README.md'),'utf8');
  const catalog = parseCatalog(markdown);
  const files=new Map();
  const assets = ['app.js','library.js','styles.css','assets/openenvision-mark.png',
    ...catalog.collections.slice(0,4).map(c=>`assets/taxonomy/${c.id}.svg`)];
  for(const file of assets) files.set(file,await readFile(resolve(root,'website',file)));
  const hash = createHash('sha256').update(catalog.sourceHash);
  for(const [name,data] of files) hash.update(name).update(data);
  const version = hash.digest('hex').slice(0,12);
  const diagramDescriptions = {
    traditional:'Two distinct patterns: CLIP-style image and text encoders align embeddings through comparison; fusion models combine modality features before task prediction.',
    mllms:'Text tokens enter a pretrained LLM. Visual features condition it through a modality interface, either as input tokens or through layer-wise attention. The dashed route is an alternative.',
    umms:'Text and visual encodings enter a common understanding-and-generation modeling framework. Text decoding and a visual decoder or generator produce the outputs. Encoders may be shared or task-specific.',
    nmms:'Early-fusion example: modality interfaces feed packed multimodal states before the first shared backbone block. Joint state evolution is an architectural property; early multimodal optimization of core weights is a separate training property. Generation is optional.'
  };
  const diagramCaptions = {
    traditional:'Embedding alignment and cross-modal feature interaction are distinct patterns.',
    mllms:'A pretrained language backbone receives perceptual evidence through a modality interface.',
    umms:'Understanding and generation share a modeling framework; encoders and decoders may differ.',
    nmms:'Fusion structure and foundation-training history are assessed separately. UMM membership may overlap.'
  };
  const replacements = {
    VERSION:version, COUNT:catalog.entries.length, SPAN:`${catalog.years.at(-1)}–${catalog.years[0]}`,
    YEARS:catalog.years.map(y=>`<option value="${y}">${y}</option>`).join(''),
    SIDEBAR:`<h3 class="sidebar-heading">Collections</h3><a class="category-button active" href="?">All entries <span>${catalog.entries.length}</span></a>`+catalog.collections.map(c=>`<a class="category-button" href="?collection=${c.id}#library">${c.label}<span>${c.count}</span></a>`).join(''),
    ENTRIES:catalog.entries.slice(0,12).map(e=>rowHTML(e)).join(''),
    TAXONOMY:catalog.collections.slice(0,4).map((c,i)=>`<article class="taxonomy-column"><div class="taxonomy-label"><span class="taxonomy-index">0${i+1}</span><h3>${c.label}</h3></div><img src="./assets/taxonomy/${c.id}.svg?v=${version}" class="taxonomy-diagram" width="300" height="360" loading="lazy" decoding="async" alt="${esc(diagramDescriptions[c.id])}"><p>${esc(diagramCaptions[c.id])}</p><a class="text-link" href="?collection=${c.id}#library" data-collection="${c.id}">Explore collection ↗</a></article>`).join(''),
  };
  let html = await readFile(resolve(root,'website/index.html'),'utf8');
  html = html.replace(/\{\{(\w+)\}\}/g,(_,key)=>{if(!(key in replacements))throw new Error(`Unknown template key: ${key}`);return replacements[key];});
  files.set('index.html',html);
  files.set('catalog.json',JSON.stringify(catalog));
  files.set('.nojekyll','');
  files.set('robots.txt','User-agent: *\nAllow: /\nSitemap: https://openenvision.github.io/Awesome-Multimodal-Modeling/sitemap.xml\n');
  files.set('sitemap.xml','<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://openenvision.github.io/Awesome-Multimodal-Modeling/</loc></url></urlset>');
  if(write) for(const [name,data] of files) {
    const destination=resolve(root,'dist',name);
    await mkdir(dirname(destination),{recursive:true});
    await writeFile(destination,data);
  }
  if(write) console.log(`Built ${catalog.entries.length} entries across ${catalog.collections.length} collections and ${catalog.sections.length} sections → dist/`);
  return {catalog,files};
}
if(process.argv[1]===fileURLToPath(import.meta.url)) await build();
                    
