import { describe, expect, it } from "vitest";

import {
  applyChange,
  countOverrides,
  NO_OVERRIDES,
  parseOverrides,
  toOverrideList,
} from "@/lib/batch/overrides";

describe("applyChange", () => {
  it("đặt nhãn sửa cho đúng dòng và không đụng dòng khác", () => {
    const step1 = applyChange(NO_OVERRIDES, 7, { sentiment: "neutral" });
    const step2 = applyChange(step1, 3, { topic: "facility" });
    expect(step2).toEqual({ 7: { sentiment: "neutral" }, 3: { topic: "facility" } });
  });

  it("giữ trường kia khi chỉ sửa một trường", () => {
    const step1 = applyChange(NO_OVERRIDES, 1, { sentiment: "negative" });
    const step2 = applyChange(step1, 1, { topic: "others" });
    expect(step2[1]).toEqual({ sentiment: "negative", topic: "others" });
  });

  it("null xoá nhãn sửa, undefined giữ nguyên", () => {
    const start = applyChange(NO_OVERRIDES, 1, { sentiment: "negative", topic: "others" });
    expect(applyChange(start, 1, { sentiment: null })[1]).toEqual({ topic: "others" });
    expect(applyChange(start, 1, { topic: undefined })[1]).toEqual({
      sentiment: "negative",
      topic: "others",
    });
  });

  it("bỏ hẳn mục khi không còn trường nào", () => {
    const start = applyChange(NO_OVERRIDES, 1, { sentiment: "negative" });
    const cleared = applyChange(start, 1, { sentiment: null });
    expect(cleared).toEqual({});
    expect(1 in cleared).toBe(false);
  });

  it("không sửa đối tượng cũ", () => {
    const start = applyChange(NO_OVERRIDES, 1, { sentiment: "negative" });
    const snapshot = JSON.stringify(start);
    applyChange(start, 1, { sentiment: "positive" });
    applyChange(start, 2, { topic: "facility" });
    expect(JSON.stringify(start)).toBe(snapshot);
    expect(NO_OVERRIDES).toEqual({});
  });
});

describe("toOverrideList", () => {
  it("sắp theo chỉ số dòng và chỉ mang trường đã sửa", () => {
    let overrides = applyChange(NO_OVERRIDES, 10, { topic: "lecturer" });
    overrides = applyChange(overrides, 2, { sentiment: "positive" });
    overrides = applyChange(overrides, 5, { sentiment: "neutral", topic: "others" });
    expect(toOverrideList(overrides)).toEqual([
      { index: 2, sentiment: "positive" },
      { index: 5, sentiment: "neutral", topic: "others" },
      { index: 10, topic: "lecturer" },
    ]);
  });

  it("rỗng khi chưa sửa gì", () => {
    expect(toOverrideList(NO_OVERRIDES)).toEqual([]);
  });
});

describe("countOverrides", () => {
  const overrides = applyChange(
    applyChange(applyChange(NO_OVERRIDES, 1, { sentiment: "positive" }), 4, {
      sentiment: "negative",
      topic: "facility",
    }),
    9,
    { topic: "others" },
  );

  it("đếm số câu, không đếm số trường", () => {
    expect(countOverrides(overrides)).toBe(3);
  });

  it("chỉ đếm trong tập chỉ số cho trước", () => {
    expect(countOverrides(overrides, new Set([1, 4, 100]))).toBe(2);
    expect(countOverrides(overrides, new Set())).toBe(0);
  });
});

describe("parseOverrides", () => {
  it("nhận dữ liệu đúng hình dạng", () => {
    expect(parseOverrides({ "3": { sentiment: "negative" }, "8": { topic: "facility" } })).toEqual({
      3: { sentiment: "negative" },
      8: { topic: "facility" },
    });
  });

  it("bỏ mục có chỉ số hoặc nhãn không hợp lệ", () => {
    const parsed = parseOverrides({
      "-1": { sentiment: "negative" },
      abc: { sentiment: "negative" },
      "1.5": { sentiment: "negative" },
      "2": { sentiment: "angry" },
      "3": { sentiment: "negative", topic: "unknown" },
      "4": null,
      "5": "negative",
    });
    expect(parsed).toEqual({ 3: { sentiment: "negative" } });
  });

  it("trả rỗng khi không phải đối tượng", () => {
    expect(parseOverrides(null)).toEqual({});
    expect(parseOverrides([])).toEqual({});
    expect(parseOverrides("x")).toEqual({});
    expect(parseOverrides(undefined)).toEqual({});
  });
});
