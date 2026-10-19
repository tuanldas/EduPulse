#!/usr/bin/env node
// Ghi fixture của phân tích hàng loạt và dashboard từ API THẬT, để test giao diện dùng đúng hình dạng và
// giá trị máy chủ trả về (không viết tay). Dữ liệu vào là 20 câu tự viết của `tests/fixtures/golden_sentences.json`
// (không có câu UIT-VSFC) cộng cột nhóm "minh hoạ" gán vòng tròn A–E, nên fixture có cả câu không dấu, câu có
// số điện thoại, câu dài chưa giải thích được và nhóm minh hoạ.
//
//   cd apps/web && FIXTURE_API_BASE=http://localhost:8000 node scripts/record-batch-fixtures.mjs
//
// Ghi vào apps/web/tests/fixtures/batch-*.json. Id batch được thay bằng giá trị cố định để lần ghi sau không
// làm đổi diff vô nghĩa.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const base = (process.env.FIXTURE_API_BASE ?? "http://localhost:8000").replace(/\/+$/, "");
const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(here, "../tests/fixtures");
const goldenPath = path.resolve(here, "../../../tests/fixtures/golden_sentences.json");

const TEXT_COLUMN = "Góp ý";
const GROUP_COLUMN = "Nhóm (minh hoạ)";
const GROUPS = ["Nhóm A", "Nhóm B", "Nhóm C", "Nhóm D", "Nhóm E"];
const FIXED_ID = "fixture-batch";
const BUDGET = 20;

function csvCell(value) {
  return `"${value.replaceAll('"', '""')}"`;
}

async function api(method, route, { json, form } = {}) {
  const response = await fetch(`${base}${route}`, {
    method,
    headers: json === undefined ? undefined : { "content-type": "application/json" },
    body: json === undefined ? form : JSON.stringify(json),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`${method} ${route} -> ${response.status}: ${text.slice(0, 300)}`);
  return text === "" ? null : JSON.parse(text);
}

async function save(name, data) {
  await writeFile(path.join(outDir, name), `${JSON.stringify(data, null, 2)}\n`, "utf-8");
  console.log("ghi", name);
}

const golden = JSON.parse(await readFile(goldenPath, "utf-8"));
const sentences = golden.items.map((item) => item.text);
const csv = [
  `${csvCell(TEXT_COLUMN)},${csvCell(GROUP_COLUMN)}`,
  ...sentences.map((text, index) => `${csvCell(text)},${csvCell(GROUPS[index % GROUPS.length])}`),
].join("\n");

await mkdir(outDir, { recursive: true });

const form = new FormData();
form.append("file", new Blob([`﻿${csv}\n`], { type: "text/csv" }), "gop-y-tu-viet.csv");
const upload = await api("POST", "/api/batches", { form });
const id = upload.batch_id;

try {
  await api("POST", `/api/batches/${id}/run`, {
    json: { text_col: TEXT_COLUMN, group_col: GROUP_COLUMN, review_budget: BUDGET },
  });
  let status;
  for (let attempt = 0; attempt < 240; attempt += 1) {
    status = await api("GET", `/api/batches/${id}`);
    if (status.state === "done" || status.state === "failed") break;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  if (status?.state !== "done") throw new Error(`batch không xong: ${JSON.stringify(status)}`);

  const results = await api("GET", `/api/batches/${id}/results`);
  const insights = await api("POST", `/api/batches/${id}/insights`, { json: { groups: null } });
  const filtered = await api("POST", `/api/batches/${id}/insights`, { json: { groups: [GROUPS[0]] } });

  const hasHighlights = (explanation) =>
    explanation.sentiment.support.length + explanation.topic.support.length > 0;
  const rowAt = (index) => results.rows.find((row) => row.index === index);
  const tooLong = results.rows.find((row) => !row.explainable);

  // Giải thích có từ được tô (ưu tiên câu trong danh sách duyệt, vốn là câu mô hình không chắc) và một giải
  // thích không tô được từ nào (câu rất chắc chắn), để test cả hai nhánh hiển thị bằng dữ liệu thật.
  let withHighlights = null;
  let noHighlights = null;
  for (const index of [...results.review_order, ...results.rows.map((row) => row.index)]) {
    if (withHighlights !== null && noHighlights !== null) break;
    if (!rowAt(index)?.explainable) continue;
    const response = await api("GET", `/api/batches/${id}/rows/${index}/explain`);
    if (hasHighlights(response)) withHighlights ??= { index, response };
    else noHighlights ??= { index, response };
  }

  await save("batch-upload.json", { ...upload, batch_id: FIXED_ID });
  await save("batch-status-done.json", status);
  await save("batch-results.json", results);
  await save("batch-explain.json", { withHighlights, noHighlights, tooLongIndex: tooLong?.index ?? null });
  await save("batch-insights.json", insights);
  await save("batch-insights-filtered.json", filtered);
} finally {
  await api("DELETE", `/api/batches/${id}`).catch(() => undefined);
}
