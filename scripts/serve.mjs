import {createServer} from 'node:http';
import {watch} from 'node:fs';
import {readFile} from 'node:fs/promises';
import {extname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build, readAssets, generatedPaths} from './build.mjs';
const root=fileURLToPath(new URL('../',import.meta.url));
const base='/Awesome-Multimodal-Modeling/';
const port=Number(process.env.PORT||4173);
const production=process.argv.includes('--production');
let files;
if(production){
  files=await readAssets();
  try {
    for(const name of [...generatedPaths,'README.md']) files.set(name,await readFile(resolve(root,name)));
  } catch(error) {
    console.error(`Missing root website files. Run npm run build first. ${error.message}`);
    process.exit(1);
  }
} else {
  files=(await build({write:false})).files;
  let timer,building=false,again=false;
  const rebuild=async()=>{
    if(building){again=true;return;}
    building=true;
    try{files=(await build({write:false})).files;console.log('Rebuilt preview. Reload the browser to see changes.');}catch(error){console.error(error);}
    finally{building=false;if(again){again=false;void rebuild();}}
  };
  const queue=()=>{clearTimeout(timer);timer=setTimeout(rebuild,120);};
  watch(resolve(root,'README.md'),queue);
  watch(resolve(root,'index.html'),queue);
  watch(resolve(root,'assets'),{recursive:true},queue);
}
const types={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.json':'application/json; charset=utf-8','.png':'image/png','.svg':'image/svg+xml','.xml':'application/xml','.txt':'text/plain; charset=utf-8'};
const server=createServer((req,res)=>{
  try{
    const url=new URL(req.url,'http://localhost');
    if(url.pathname==='/'||url.pathname===base.slice(0,-1)){res.writeHead(302,{Location:base+url.search});res.end();return;}
    if(!url.pathname.startsWith(base)){res.writeHead(404);res.end('Not found');return;}
    const file=decodeURIComponent(url.pathname.slice(base.length))||'index.html';
    if(!files.has(file)){res.writeHead(404);res.end('Not found');return;}
    const data=files.get(file);
    res.writeHead(200,{'Content-Type':types[extname(file)]||'application/octet-stream','Cache-Control':'no-cache','X-Content-Type-Options':'nosniff'});
    res.end(data);
  }catch{res.writeHead(400);res.end('Bad request');}
});
server.on('error',error=>{console.error(`Preview server: ${error.message}`);process.exit(1);});
server.listen(port,'127.0.0.1',()=>console.log(`Preview: http://127.0.0.1:${port}${base}`));
