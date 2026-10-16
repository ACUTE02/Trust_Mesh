// Prints raw and gzip size of every built JS/CSS/font file in dist/assets, grouped. Run after `npm run build`.
// Usage (from frontend/): node ../scripts/bundle_sizes.mjs
import { readdirSync, readFileSync, statSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { join } from "node:path";

const dir = join(process.cwd(), "dist", "assets");
const rows = readdirSync(dir).map((f) => {
  const buf = readFileSync(join(dir, f));
  const kind = f.endsWith(".js") ? "js" : f.endsWith(".css") ? "css" : f.endsWith(".woff2") ? "font" : "other";
  return { f, kind, raw: statSync(join(dir, f)).size, gz: kind === "font" ? statSync(join(dir, f)).size : gzipSync(buf).length };
});
const kb = (n) => (n / 1024).toFixed(1).padStart(8) + " KB";
for (const kind of ["js", "css", "font"]) {
  console.log(`\n${kind.toUpperCase()} (gzip column; fonts are already compressed)`);
  for (const r of rows.filter((x) => x.kind === kind).sort((a, b) => b.gz - a.gz)) console.log(`${kb(r.gz)}  ${r.f}`);
  console.log(`${kb(rows.filter((x) => x.kind === kind).reduce((s, r) => s + r.gz, 0))}  total`);
}
