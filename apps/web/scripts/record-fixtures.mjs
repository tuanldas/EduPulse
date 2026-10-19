#!/usr/bin/env node
// Ghi lại phản hồi THẬT của API vào tests/fixtures/ để test giao diện dùng, không viết tay dữ liệu.
//
//   cd apps/web && FIXTURE_API_BASE=http://localhost:8000 pnpm record:fixtures
//
// Câu đầu vào là 20 câu tự viết của repo (tests/fixtures/golden_sentences.json ở thư mục gốc), không
// phải câu UIT-VSFC. `ui_samples` của /api/config trên máy dev đến từ dữ liệu demo (có câu UIT-VSFC)
// nên được thay bằng câu tự viết trước khi ghi. model-card chỉ chứa số tổng hợp.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const base = (process.env.FIXTURE_API_BASE ?? "http://localhost:8000").replace(/\/+$/, "");
const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, "..", "tests", "fixtures");
const goldenPath = path.join(here, "..", "..", "..", "tests", "fixtures", "golden_sentences.json");

const golden = JSON.parse(await readFile(goldenPath, "utf-8")).items;
const sentence = (id) => {
  const item = golden.find((candidate) => candidate.id === id);
  if (!item) throw new Error(`không có câu ${id} trong golden_sentences.json`);
  return item.text;
};

async function call(route, init) {
  const response = await fetch(`${base}${route}`, {
    ...init,
    headers: { "content-type": "application/json" },
  });
  const body = await response.json();
  return { status: response.status, body };
}

async function save(name, value) {
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, name), `${JSON.stringify(value, null, 2)}\n`, "utf-8");
  console.log(`ghi ${name}`);
}

const post = (route, payload) => call(route, { method: "POST", body: JSON.stringify(payload) });

// g02 phân tích thường · g08 giải thích có 3 từ ủng hộ + 1 từ phản đối · g13 không dấu · g14 số điện
// thoại · g15 tên · g17 câu dài (> giới hạn từ giải thích).
for (const id of ["g02", "g08", "g13", "g14", "g15", "g17"]) {
  const { status, body } = await post("/api/analyze", { text: sentence(id), compare: false });
  if (status !== 200) throw new Error(`analyze ${id}: HTTP ${status}`);
  await save(`analyze-${id}.json`, { text: sentence(id), response: body });
}

// g02: không có từ nào đủ lớn để tô (cả hai đầu ra); g08: có từ ủng hộ và phản đối.
for (const id of ["g02", "g08"]) {
  const explained = await post("/api/explain", { text: sentence(id) });
  if (explained.status !== 200) throw new Error(`explain ${id}: HTTP ${explained.status}`);
  await save(`explain-${id}.json`, { text: sentence(id), response: explained.body });
}

// Mô hình cơ sở: ghi nguyên trạng, kể cả khi máy chủ chưa làm xong (501 not_implemented).
const compare = await post("/api/analyze", { text: sentence("g02"), compare: true });
await save("analyze-g02-compare.json", { text: sentence("g02"), status: compare.status, body: compare.body });

const config = await call("/api/config");
if (config.status !== 200) throw new Error(`config: HTTP ${config.status}`);
config.body.ui_samples = [
  { text: sentence("g01"), no_diacritics: false },
  { text: sentence("g02"), no_diacritics: false },
  { text: sentence("g13"), no_diacritics: true },
];
await save("config.json", config.body);

const card = await call("/api/model-card");
if (card.status !== 200) throw new Error(`model-card: HTTP ${card.status}`);
await save("model-card.json", card.body);
