import { spawn, execFileSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { once } from "node:events";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { createServer } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..");
const song = resolve(repo, "songs/ievan/allvoice");
const inputPath = resolve(process.argv[2] ?? resolve(song, "visualizer_input.json"));
const audioPath = resolve(process.argv[3] ?? resolve(song, "master.mp3"));
const videoPath = resolve(process.argv[4] ?? resolve(song, "喉モデル_イエヴァンポルッカ_全部声_喉の動き.mp4"));
const outDir = resolve(dirname(videoPath), "visualization");
const chromePath = process.env.CHROME_BIN ?? ["chromium", "chromium-browser", "google-chrome"].map(name => { try { return execFileSync("which",[name],{encoding:"utf8"}).trim(); } catch { return ""; } }).find(Boolean);
const fps = 30;
if (existsSync(videoPath)) throw new Error(`Refusing to overwrite video: ${videoPath}`);

if (!existsSync(inputPath) || !existsSync(audioPath)) throw new Error("Usage: node render.mjs [visualizer_input.json] [master.mp3] [output.mp4]");
if (!chromePath || !existsSync(chromePath)) throw new Error(`Chromium binary not found: ${chromePath}`);
mkdirSync(outDir, { recursive: true });
process.env.CHIMERA_MULTI_INPUT = inputPath;

const server = await createServer({ configFile: resolve(here, "vite.config.ts"), clearScreen: false });
await server.listen();
const address = server.httpServer.address();
const port = typeof address === "object" && address ? address.port : 4178;
const browser = await chromium.launch({ executablePath: chromePath, headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage", "--font-render-hinting=none"] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
page.on("pageerror", (error) => process.stderr.write(`pageerror: ${error.stack ?? error.message}\n`));
page.on("console", (message) => { if (message.type() === "error") process.stderr.write(`console: ${message.text()}\n`); });
try {
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.chimeraReady === true);
  await page.evaluate(async () => {
    const response = await fetch("/__allvoice_input.json");
    if (!response.ok) throw new Error(`Input fetch failed: ${response.status}`);
    await window.chimeraMulti.loadInput(await response.json());
  });
  const dimensions = await page.evaluate(() => window.chimeraMulti.dimensions());
  const validation = await page.evaluate(() => window.chimeraMulti.validate());
  writeFileSync(resolve(outDir, "state_validation.json"), `${JSON.stringify(validation, null, 2)}\n`);
  const comparison = await page.evaluate(() => window.chimeraMulti.pixelComparison());
  const imageFromDataUrl = (url) => Buffer.from(url.slice(url.indexOf(",") + 1), "base64");
  writeFileSync(resolve(outDir, "canonical_source_frame.png"), imageFromDataUrl(comparison.source));
  writeFileSync(resolve(outDir, "single_throat_frame.png"), imageFromDataUrl(comparison.single));
  writeFileSync(resolve(outDir, "pixel_comparison.png"), imageFromDataUrl(comparison.visual));
  const pixelReport = {
    method: "Pixel-by-pixel comparison of the unmodified CHIMERA AnatomyRenderer + Glow in a single-throat source frame and the same renderer path used by the multi-throat visualizer.",
    comparison_seconds: comparison.seconds, throat: `${comparison.part.name} / throat ${comparison.part.throat}`, part_id: comparison.part.id,
    source_renderer: "vendor/chimera/src/ui/render/anatomy.ts",
    glow_renderer: "vendor/chimera/src/ui/render/vector.ts :: Glow", dimensions: { width: comparison.stats.width, height: comparison.stats.height },
    runtime_adapter: "Saved solver areas, measured grids, learned body and source area/flow; original VoiceModel constructor with a replay-engine adapter versus replay model. WASM is unavailable; this is not a comparison against an independently running full original app.",
    metrics: comparison.stats,
    files: { source: "canonical_source_frame.png", single: "single_throat_frame.png", comparison: "pixel_comparison.png" },
  };
  writeFileSync(resolve(outDir, "pixel_comparison.json"), `${JSON.stringify(pixelReport, null, 2)}\n`);

  const ffmpeg = spawn("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y",
    "-f", "image2pipe", "-vcodec", "png", "-framerate", String(fps), "-i", "pipe:0",
    "-i", audioPath, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast",
    "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p", "-r", String(fps), "-c:a", "aac", "-b:a", "192k",
    "-shortest", "-movflags", "+faststart", videoPath], { stdio: ["pipe", "ignore", "pipe"] });
  let ffmpegErrors = "";
  ffmpeg.stderr.setEncoding("utf8");
  ffmpeg.stderr.on("data", (chunk) => { ffmpegErrors += chunk; });
  const pngPipe = ffmpeg.stdin;
  const frames = Math.ceil(dimensions.audio_seconds * fps);
  await page.evaluate(() => window.chimeraMulti.reset());
  for (let frame = 0; frame < frames; frame++) {
    const time = frame / fps;
    const png = await page.evaluate((t) => { window.chimeraMulti.drawAt(t); return window.chimeraMulti.capture(); }, time);
    if (!pngPipe.write(imageFromDataUrl(png))) await once(pngPipe, "drain");
    if (frame === 300 || frame === Math.min(frames-1,150)) writeFileSync(resolve(outDir, `preview_${frame}.png`), imageFromDataUrl(png));
    if ((frame + 1) % 60 === 0 || frame + 1 === frames) process.stdout.write(`rendered ${frame + 1}/${frames} frames\n`);
  }
  pngPipe.end();
  const [exitCode] = await once(ffmpeg, "close");
  if (exitCode !== 0) throw new Error(`ffmpeg exited ${exitCode}: ${ffmpegErrors}`);
  const videoReport = { file: videoPath, width: dimensions.width, height: dimensions.height, fps,
    frames, duration_seconds: frames / fps, audio_source: audioPath, audio_source_sha256: null };
  const { createHash } = await import("node:crypto");
  videoReport.audio_source_sha256 = createHash("sha256").update(readFileSync(audioPath)).digest("hex");
  writeFileSync(resolve(outDir, "visualizer_render_report.json"), `${JSON.stringify({ ...videoReport, pixel_comparison: pixelReport }, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ ...videoReport, pixel_metrics: comparison.stats })}\n`);
} finally {
  await page.close();
  await browser.close();
  await server.close();
}
