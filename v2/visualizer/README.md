# Multiple-throat saved-state visualization

From this directory: `npm ci --ignore-scripts` then `npm run build` (Node.js/npm). The lockfile is included. Neither node_modules nor build products are committed. `publicDir` contains only the selected fonts and their notices; no upstream dictionary, shows, TSP or CSD data is copied.

Install Chromium/Google Chrome and ffmpeg separately. `CHROME_BIN` may name the browser executable; otherwise it is discovered on PATH. Run `node render.mjs input.json audio.mp3 output.mp4`, or use songkit from v2. The compositor burns the CHIMERA copyright credit into every video frame at bottom right.

The vendored drawing source is byte-identical to CHIMERA commit 092b16ce12b927d473c884ca2dfae53ce1b995df. See `vendor/chimera/PROVENANCE.md`, `vendor/chimera/LICENSE` and the root THIRD_PARTY_NOTICES.md. Vite replaces the live app bindings and extracts TractMap for saved-state playback. Pixel comparison uses the same saved state through two renderer paths; it is not an independent test against the complete original WASM app. Build artifacts retain the font licenses; include the root third-party notices when redistributing them.
