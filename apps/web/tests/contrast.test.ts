import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { TEXT_ON_SOLID } from "@/lib/colors";

import { contrast, over, parseColor, type Rgba } from "./helpers/contrast";

// Canh tương phản WCAG 2.1 AA của các cặp token trong app/globals.css ở cả hai theme: chữ ≥ 4,5:1,
// viền và nền biểu đồ ≥ 3:1 (docs/design-guidelines.md mục 2 và 7). Đọc trực tiếp từ file CSS nên
// sửa token mà làm tụt tương phản sẽ làm test đỏ.
const CSS = readFileSync(path.join(import.meta.dirname, "..", "src", "app", "globals.css"), "utf-8");

type Theme = "light" | "dark";

function readBlock(selector: string): Record<string, string> {
  const start = CSS.search(new RegExp(`^${selector.replace(".", "\\.")}\\s*\\{`, "m"));
  expect(start, `không thấy khối ${selector}`).toBeGreaterThanOrEqual(0);
  const body = CSS.slice(start, CSS.indexOf("\n}", start));
  return Object.fromEntries([...body.matchAll(/--([a-z0-9-]+):\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]));
}

const VARS: Record<Theme, Record<string, string>> = {
  light: readBlock(":root"),
  dark: readBlock(".dark"),
};

/** Màu của token ở một theme; token chỉ khai ở :root (nền đặc dùng chung hai theme) lấy từ theme sáng. */
function colorOf(theme: Theme, token: string): Rgba {
  const value = VARS[theme][token] ?? VARS.light[token];
  expect(value, `thiếu token --${token}`).toBeDefined();
  return parseColor(value);
}

const TONES = ["neg", "neu", "pos", "gv", "ct", "csvc", "khac", "flag"] as const;
const THEMES: Theme[] = ["light", "dark"];

describe.each(THEMES)("tương phản token · theme %s", (theme) => {
  const ratio = (fg: string, bg: string) => contrast(colorOf(theme, fg), colorOf(theme, bg));

  it("chữ thường và chữ phụ đọc được trên nền trang và nền bg2", () => {
    expect(ratio("foreground", "background")).toBeGreaterThanOrEqual(4.5);
    expect(ratio("foreground", "muted")).toBeGreaterThanOrEqual(4.5);
    expect(ratio("muted-foreground", "background")).toBeGreaterThanOrEqual(4.5);
    expect(ratio("muted-foreground", "muted")).toBeGreaterThanOrEqual(4.5);
  });

  it("liên kết và chữ màu primary đọc được (tối dùng màu sáng hơn nền nút)", () => {
    expect(ratio("link", "background")).toBeGreaterThanOrEqual(4.5);
    expect(ratio("primary-text", "background")).toBeGreaterThanOrEqual(4.5);
    expect(ratio("primary-text", "muted")).toBeGreaterThanOrEqual(4.5);
    expect(ratio("primary-text", "primary-soft")).toBeGreaterThanOrEqual(4.5);
  });

  it("nút primary: chữ trắng trên nền nút (cả khi hover), và nền nút nhìn rõ trên nền trang", () => {
    expect(ratio("primary-foreground", "primary")).toBeGreaterThanOrEqual(4.5);
    expect(ratio("primary-foreground", "primary-hover")).toBeGreaterThanOrEqual(4.5);
    expect(ratio("primary", "background")).toBeGreaterThanOrEqual(3);
    expect(ratio("primary-hover", "background")).toBeGreaterThanOrEqual(3);
  });

  it("viền ô nhập và khung nhìn rõ trên nền trang (WCAG 1.4.11)", () => {
    expect(ratio("border", "background")).toBeGreaterThanOrEqual(3);
  });

  it("vòng tiêu điểm bàn phím (ring-ring/70 trong component) nhìn rõ trên nền trang và nền bg2", () => {
    const [r, g, b] = colorOf(theme, "ring");
    for (const surface of ["background", "muted"]) {
      const ring = over([r, g, b, 0.7], colorOf(theme, surface));
      expect(contrast(ring, colorOf(theme, surface))).toBeGreaterThanOrEqual(3);
    }
    // Viền tiêu điểm mặc định của trình duyệt dùng màu ring đặc.
    expect(ratio("ring", "background")).toBeGreaterThanOrEqual(4.5);
  });

  it.each(["info", "warn", "ok", "err"])("khối thông báo %s: chữ trên nền nhạt", (tone) => {
    expect(ratio(`${tone}-fg`, `${tone}-bg`)).toBeGreaterThanOrEqual(4.5);
  });

  it.each(TONES)("badge %s: chữ trên nền badge, và chữ/icon trên nền trang", (tone) => {
    expect(ratio(`${tone}-fg`, `${tone}-bg`)).toBeGreaterThanOrEqual(4.5);
    expect(ratio(`${tone}-fg`, "background")).toBeGreaterThanOrEqual(4.5);
  });

  it.each(["neg", "neu", "pos", "gv", "ct", "csvc", "khac"] as const)("màu nền biểu đồ %s nhìn rõ trên nền trang", (tone) => {
    expect(ratio(tone, "background")).toBeGreaterThanOrEqual(3);
  });

  it.each(TONES)("chữ chọn sẵn trên nền đặc %s (TEXT_ON_SOLID) đọc được", (tone) => {
    const textClass = TEXT_ON_SOLID[tone];
    const text = textClass === "text-white" ? parseColor("#ffffff") : colorOf(theme, "ink");
    // `khac` (#66837A) không có màu chữ nào đạt 4,5:1; chữ tối cho 4,35:1 là mức tốt nhất, nên chỉ đặt chữ lớn lên đó.
    const needed = tone === "khac" ? 4.3 : 4.5;
    expect(contrast(text, colorOf(theme, tone))).toBeGreaterThanOrEqual(needed);
  });

  it.each(["xai-3", "xai-2", "xai-1", "xai-opp"])("chữ đọc được trên nền tô giải thích %s", (token) => {
    const page = colorOf(theme, "background");
    const highlighted = over(colorOf(theme, token), page);
    expect(contrast(colorOf(theme, "foreground"), highlighted)).toBeGreaterThanOrEqual(4.5);
  });
});
