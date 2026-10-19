import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

// Canh tương phản WCAG 2.1 AA của các cặp token trong app/globals.css ở cả hai theme: chữ ≥ 4,5:1,
// viền và nền biểu đồ ≥ 3:1 (docs/design-guidelines.md mục 2 và 7). Đọc trực tiếp từ file CSS nên
// sửa token mà làm tụt tương phản sẽ làm test đỏ.
const CSS = readFileSync(path.join(import.meta.dirname, "..", "src", "app", "globals.css"), "utf-8");

type Rgba = [number, number, number, number];
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

function parse(value: string): Rgba {
  if (value.startsWith("#")) {
    return [
      parseInt(value.slice(1, 3), 16),
      parseInt(value.slice(3, 5), 16),
      parseInt(value.slice(5, 7), 16),
      1,
    ];
  }
  const match = /^rgba\(([^)]+)\)$/.exec(value);
  if (!match) throw new Error(`không đọc được màu: ${value}`);
  const [r, g, b, a] = match[1].split(",").map(Number);
  return [r, g, b, a];
}

/** Màu của token ở một theme; token chỉ khai ở :root (nền đặc dùng chung hai theme) lấy từ theme sáng. */
function colorOf(theme: Theme, token: string): Rgba {
  const value = VARS[theme][token] ?? VARS.light[token];
  expect(value, `thiếu token --${token}`).toBeDefined();
  return parse(value);
}

function over(fg: Rgba, bg: Rgba): Rgba {
  const [r, g, b, a] = fg;
  return [a * r + (1 - a) * bg[0], a * g + (1 - a) * bg[1], a * b + (1 - a) * bg[2], 1];
}

function luminance([r, g, b]: Rgba): number {
  const channel = (x: number) => {
    const c = x / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function contrast(a: Rgba, b: Rgba): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
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

  it("nút primary: chữ trắng trên nền nút, và nền nút nhìn rõ trên nền trang", () => {
    expect(ratio("primary-foreground", "primary")).toBeGreaterThanOrEqual(4.5);
    expect(ratio("primary", "background")).toBeGreaterThanOrEqual(3);
  });

  it("viền ô nhập và khung nhìn rõ trên nền trang (WCAG 1.4.11)", () => {
    expect(ratio("border", "background")).toBeGreaterThanOrEqual(3);
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

  it.each(["xai-3", "xai-2", "xai-1", "xai-opp"])("chữ đọc được trên nền tô giải thích %s", (token) => {
    const page = colorOf(theme, "background");
    const highlighted = over(colorOf(theme, token), page);
    expect(contrast(colorOf(theme, "foreground"), highlighted)).toBeGreaterThanOrEqual(4.5);
  });
});
