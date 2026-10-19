import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { NO_OVERRIDES } from "@/lib/batch/overrides";
import {
  changeLabel,
  discardReviewEdits,
  hasUnexportedEdits,
  markExported,
  NO_EDITS,
  parseReviewEdits,
  useReviewEdits,
} from "@/lib/batch/review-edits";

describe("trạng thái sửa nhãn", () => {
  it("chưa sửa gì thì không có gì để mất", () => {
    expect(hasUnexportedEdits(NO_EDITS)).toBe(false);
  });

  it("sửa nhãn mà chưa xuất thì còn chưa xuất; xuất rồi thì hết", () => {
    const edited = changeLabel(NO_EDITS, 3, { sentiment: "negative" });
    expect(hasUnexportedEdits(edited)).toBe(true);
    const exported = markExported(edited);
    expect(hasUnexportedEdits(exported)).toBe(false);
    expect(exported.overrides).toEqual(edited.overrides);
  });

  it("sửa thêm sau khi xuất thì lại còn chưa xuất", () => {
    const exported = markExported(changeLabel(NO_EDITS, 3, { sentiment: "negative" }));
    const edited = changeLabel(exported, 4, { topic: "facility" });
    expect(hasUnexportedEdits(edited)).toBe(true);
  });

  it("xoá hết nhãn đã sửa thì không còn gì để mất", () => {
    const edited = changeLabel(NO_EDITS, 3, { sentiment: "negative" });
    const cleared = changeLabel(edited, 3, { sentiment: null });
    expect(cleared.overrides).toEqual(NO_OVERRIDES);
    expect(hasUnexportedEdits(cleared)).toBe(false);
  });
});

describe("parseReviewEdits", () => {
  it("đọc dữ liệu đúng và kẹp exportedRev không vượt rev", () => {
    expect(
      parseReviewEdits({ overrides: { "2": { topic: "others" } }, rev: 3, exportedRev: 9 }),
    ).toEqual({ overrides: { 2: { topic: "others" } }, rev: 3, exportedRev: 3 });
  });

  it("rơi về trạng thái rỗng khi dữ liệu hỏng", () => {
    expect(parseReviewEdits("x")).toBe(NO_EDITS);
    expect(parseReviewEdits(null)).toBe(NO_EDITS);
    expect(parseReviewEdits({ overrides: 5, rev: -1, exportedRev: "a" })).toEqual({
      overrides: {},
      rev: 0,
      exportedRev: 0,
    });
  });
});

describe("useReviewEdits", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("lưu vào sessionStorage theo batch và đọc lại được ở lần dựng sau", () => {
    const first = renderHook(() => useReviewEdits("batch-a"));
    act(() => first.result.current.change(5, { sentiment: "neutral" }));
    expect(first.result.current.count).toBe(1);
    expect(first.result.current.unexported).toBe(true);
    first.unmount();

    const second = renderHook(() => useReviewEdits("batch-a"));
    expect(second.result.current.edits.overrides).toEqual({ 5: { sentiment: "neutral" } });
    expect(second.result.current.unexported).toBe(true);

    act(() => second.result.current.markExported());
    expect(second.result.current.unexported).toBe(false);
    expect(second.result.current.count).toBe(1);
    discardReviewEdits("batch-a");
  });

  it("tách riêng từng batch", () => {
    const a = renderHook(() => useReviewEdits("batch-x"));
    const b = renderHook(() => useReviewEdits("batch-y"));
    act(() => a.result.current.change(1, { topic: "facility" }));
    expect(a.result.current.count).toBe(1);
    expect(b.result.current.count).toBe(0);
    discardReviewEdits("batch-x");
    discardReviewEdits("batch-y");
  });

  it("hai nơi dùng cùng batch thấy cùng một trạng thái", () => {
    const a = renderHook(() => useReviewEdits("batch-z"));
    const b = renderHook(() => useReviewEdits("batch-z"));
    act(() => a.result.current.change(2, { sentiment: "positive" }));
    expect(b.result.current.edits.overrides).toEqual({ 2: { sentiment: "positive" } });
    discardReviewEdits("batch-z");
  });

  it("không có batch thì bỏ qua mọi thao tác ghi", () => {
    const { result } = renderHook(() => useReviewEdits(null));
    act(() => result.current.change(1, { sentiment: "negative" }));
    expect(result.current.count).toBe(0);
    expect(result.current.unexported).toBe(false);
  });

  it("discardReviewEdits xoá nhãn đã sửa", () => {
    const { result } = renderHook(() => useReviewEdits("batch-d"));
    act(() => result.current.change(1, { sentiment: "negative" }));
    act(() => discardReviewEdits("batch-d"));
    expect(result.current.count).toBe(0);
    expect(window.sessionStorage.getItem("edupulse.edits.batch-d")).toBeNull();
  });

  it("chịu được dữ liệu hỏng trong sessionStorage", () => {
    window.sessionStorage.setItem("edupulse.edits.batch-bad", "{không phải json");
    const { result } = renderHook(() => useReviewEdits("batch-bad"));
    expect(result.current.edits).toBe(NO_EDITS);
  });
});
