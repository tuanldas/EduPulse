import { describe, expect, it } from "vitest";

import { contrast, parseColor } from "./helpers/contrast";

import {
  DELTA_SCALE,
  SEQUENTIAL_SCALE,
  TONE_CLASS,
  deltaColor,
  deltaStep,
  sequentialColor,
  sequentialStep,
  toneVar,
} from "@/lib/colors";

describe("thang tuần tự (confusion matrix)", () => {
  it("chia tỉ lệ thành mười bậc đều nhau, 0 vẫn là bậc 1", () => {
    expect(sequentialStep(0)).toBe(1);
    expect(sequentialStep(0.1)).toBe(1);
    expect(sequentialStep(0.101)).toBe(2);
    expect(sequentialStep(0.22)).toBe(3);
    expect(sequentialStep(0.534)).toBe(6);
    expect(sequentialStep(0.974)).toBe(10);
    expect(sequentialStep(1)).toBe(10);
  });

  it("kẹp giá trị ngoài khoảng và bỏ qua giá trị không hợp lệ", () => {
    expect(sequentialStep(-1)).toBe(1);
    expect(sequentialStep(7)).toBe(10);
    expect(sequentialStep(Number.NaN)).toBe(1);
  });

  it("chữ tối ở bậc 1–7, chữ trắng ở bậc 8–10", () => {
    expect(sequentialColor(0.534)).toEqual({ background: SEQUENTIAL_SCALE[5], color: "#14171E" });
    expect(sequentialColor(0.7).color).toBe("#14171E");
    expect(sequentialColor(0.71).color).toBe("#FFFFFF");
    expect(sequentialColor(0.974)).toEqual({ background: SEQUENTIAL_SCALE[9], color: "#FFFFFF" });
  });
});

describe("thang chênh so với mức nền (heatmap)", () => {
  it("chia sáu bậc theo điểm phần trăm", () => {
    expect(deltaStep(-3)).toBe(0);
    expect(deltaStep(0)).toBe(0);
    expect(deltaStep(3)).toBe(1);
    expect(deltaStep(5)).toBe(2);
    expect(deltaStep(14.9)).toBe(3);
    expect(deltaStep(19.99)).toBe(4);
    expect(deltaStep(20)).toBe(5);
    expect(deltaStep(80)).toBe(5);
    expect(deltaStep(Number.NaN)).toBe(0);
  });

  it("chỉ bậc cao nhất dùng chữ trắng", () => {
    expect(deltaColor(3)).toEqual({ background: DELTA_SCALE[1], color: "#14171E" });
    expect(deltaColor(25)).toEqual({ background: DELTA_SCALE[5], color: "#FFFFFF" });
  });
});

describe("khoá màu ngữ nghĩa", () => {
  it("mỗi khoá có lớp badge, chữ và nền đặc", () => {
    for (const [tone, classes] of Object.entries(TONE_CLASS)) {
      expect(classes.badge).toContain(`bg-${tone}-bg`);
      expect(classes.badge).toContain(`text-${tone}-fg`);
      expect(classes.text).toBe(`text-${tone}-fg`);
      expect(classes.solid).toBe(`bg-${tone}`);
    }
  });

  it("trả biến CSS cho SVG và style nội tuyến", () => {
    expect(toneVar("neg")).toBe("var(--neg)");
  });
});

describe("chữ trên các thang tường minh luôn đạt 4,5:1", () => {
  it("confusion matrix: mọi bậc của thang tuần tự", () => {
    for (let step = 1; step <= SEQUENTIAL_SCALE.length; step += 1) {
      const { background, color } = sequentialColor(step / SEQUENTIAL_SCALE.length);
      expect(contrast(parseColor(color), parseColor(background)), `bậc ${step}`).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("heatmap: mọi bậc của thang chênh so với mức nền", () => {
    for (const points of [-5, 2, 7, 12, 17, 25]) {
      const { background, color } = deltaColor(points);
      expect(contrast(parseColor(color), parseColor(background)), `${points} điểm`).toBeGreaterThanOrEqual(4.5);
    }
  });
});
