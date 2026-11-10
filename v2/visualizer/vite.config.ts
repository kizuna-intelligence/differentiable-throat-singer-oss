import { createReadStream, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL(".", import.meta.url));
const repo = resolve(root, "..");
const thirdParty = resolve(root, "vendor/chimera");

export default defineConfig({
  root,
  base: "/",
  publicDir: resolve(thirdParty, "public"),
  server: { host: "127.0.0.1", port: 0, strictPort: false, hmr: false, watch: null, fs: { allow: [repo, thirdParty] } },
  esbuild: { jsx: 'automatic', jsxImportSource: 'virtual:replay' },
  plugins: [{
    name: 'upstream-offline-bindings', enforce: 'pre',
    resolveId(id, importer) {
      if (id === 'virtual:tract-map') return '\0virtual:tract-map';
      if (id === 'virtual:replay/jsx-runtime' || id === 'virtual:replay/jsx-dev-runtime') return resolve(root,'src/replay_hooks.ts');
      if (importer?.includes('vendor/chimera/src/ui/components/')) {
        if (['react','../usePhosphor.ts','../../audio/AudioEngine.ts','../store.ts','../useModel.ts'].includes(id)) return resolve(root,'src/replay_hooks.ts');
      }
      if (importer?.endsWith('/ui/analysis.ts') && id === '../wasm/engine.ts') return resolve(root,'src/replay_engine.ts');
    },
    load(id) {
      if(id === '\0virtual:tract-map') {
        const source=readFileSync(resolve(thirdParty,'src/wasm/engine.ts'),'utf8');
        const a=source.indexOf('export class TractMap {');
        const b=source.indexOf('/** The main thread',a);
        // Original class body verbatim; erase only its compile-time type annotations via Vite.
        return {code: source.slice(a,b).replace('readonly scale: number','scale').replace('private readonly ph: number','ph').replace('body: HumanBody','body').replaceAll('x: number, y: number','x, y').replaceAll(': { x: number; y: number }',''),map:null};
      }
    }
  },{
    name: "allvoice-input-file",
    configureServer(server) {
      server.middlewares.use("/__allvoice_input.json", (req, res) => {
        const input = process.env.CHIMERA_MULTI_INPUT;
        if (!input) { res.statusCode = 404; res.end("CHIMERA_MULTI_INPUT is unset"); return; }
        res.setHeader("content-type", "application/json; charset=utf-8");
        createReadStream(input).pipe(res);
      });
    },
  }],
});
