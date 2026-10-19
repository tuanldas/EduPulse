import { describe, expect, it } from "vitest";

import { stepLineStates } from "@/lib/batch/run-steps";
import type { BatchStatus } from "@/lib/batch/types";

function status(overrides: Partial<BatchStatus>): BatchStatus {
  return {
    state: "running",
    step: null,
    done: 0,
    total: 100,
    progress: 0,
    eta_s: null,
    n_rows: 100,
    pii_rows: null,
    error_code: null,
    illustrative_groups: false,
    ...overrides,
  };
}

describe("stepLineStates", () => {
  it("chưa có trạng thái hoặc đang xếp hàng: mọi dòng chờ", () => {
    const pending = { pii: "pending", segment: "pending", predict: "pending", flag: "pending" };
    expect(stepLineStates(null)).toEqual(pending);
    expect(stepLineStates(status({ state: "queued" }))).toEqual(pending);
  });

  it("prepare chạy: cả hai dòng ẩn thông tin cá nhân và tách từ cùng đang chạy", () => {
    expect(stepLineStates(status({ step: "prepare" }))).toEqual({
      pii: "active",
      segment: "active",
      predict: "pending",
      flag: "pending",
    });
  });

  it("prepare xong thì đánh dấu xong cả hai dòng tương ứng", () => {
    expect(stepLineStates(status({ step: "predict" }))).toEqual({
      pii: "done",
      segment: "done",
      predict: "active",
      flag: "pending",
    });
  });

  it("flag và explain cùng thuộc dòng cuối", () => {
    const expected = { pii: "done", segment: "done", predict: "done", flag: "active" };
    expect(stepLineStates(status({ step: "flag" }))).toEqual(expected);
    expect(stepLineStates(status({ step: "explain" }))).toEqual(expected);
  });

  it("xong: mọi dòng hoàn tất", () => {
    expect(stepLineStates(status({ state: "done", step: null, progress: 1 }))).toEqual({
      pii: "done",
      segment: "done",
      predict: "done",
      flag: "done",
    });
  });

  it("lỗi giữa chừng: dòng đang chạy thành lỗi, các dòng trước giữ nguyên là xong", () => {
    expect(
      stepLineStates(status({ state: "failed", step: "predict", error_code: "internal_error" })),
    ).toEqual({ pii: "done", segment: "done", predict: "failed", flag: "pending" });
  });

  it("lỗi trước khi vào bước nào: mọi dòng chờ", () => {
    expect(
      stepLineStates(status({ state: "failed", step: null, error_code: "internal_error" })),
    ).toEqual({ pii: "pending", segment: "pending", predict: "pending", flag: "pending" });
  });
});
