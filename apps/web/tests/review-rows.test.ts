import { describe, expect, it } from "vitest";

import { pageOfPosition, paginate, reasonDetail, reasonText, rowsForTab } from "@/lib/batch/review-rows";

import { lowConfidence, makeResults, makeRow } from "./helpers/batch-rows";

describe("rowsForTab", () => {
  const results = makeResults();

  it("danh sách duyệt theo đúng thứ tự cố định của máy chủ, không theo thứ tự trong file", () => {
    const rows = rowsForTab({ ...results, review_order: [3, 0, 2] }, "review");
    expect(rows.map((row) => row.index)).toEqual([3, 0, 2]);
  });

  it("danh sách duyệt không chứa dòng ngoài hàng đợi", () => {
    expect(rowsForTab(results, "review").map((row) => row.index)).toEqual([0, 2, 3]);
  });

  it("bỏ qua chỉ số không có dòng tương ứng thay vì vỡ", () => {
    const rows = rowsForTab({ ...results, review_order: [0, 99, 2] }, "review");
    expect(rows.map((row) => row.index)).toEqual([0, 2]);
  });

  it("tab tất cả giữ thứ tự trong file", () => {
    expect(rowsForTab(results, "all").map((row) => row.index)).toEqual([0, 1, 2, 3]);
  });
});

describe("paginate", () => {
  const items = Array.from({ length: 120 }, (_, i) => i);

  it("chia 50 dòng mỗi trang", () => {
    const first = paginate(items, 1);
    expect(first.items).toHaveLength(50);
    expect(first).toMatchObject({ page: 1, pageCount: 3, total: 120, first: 1, last: 50 });
    const last = paginate(items, 3);
    expect(last.items).toEqual(items.slice(100));
    expect(last).toMatchObject({ page: 3, first: 101, last: 120 });
  });

  it("kẹp trang ngoài phạm vi", () => {
    expect(paginate(items, 0).page).toBe(1);
    expect(paginate(items, -4).page).toBe(1);
    expect(paginate(items, 99).page).toBe(3);
    expect(paginate(items, Number.NaN).page).toBe(1);
    expect(paginate(items, 2.9).page).toBe(2);
  });

  it("danh sách rỗng vẫn có 1 trang", () => {
    expect(paginate([], 1)).toEqual({
      items: [],
      page: 1,
      pageCount: 1,
      total: 0,
      first: 0,
      last: 0,
    });
  });

  it("đúng 50 dòng là 1 trang", () => {
    expect(paginate(items.slice(0, 50), 1).pageCount).toBe(1);
    expect(paginate(items.slice(0, 51), 1).pageCount).toBe(2);
  });

  it("trang chứa một vị trí", () => {
    expect(pageOfPosition(0)).toBe(1);
    expect(pageOfPosition(49)).toBe(1);
    expect(pageOfPosition(50)).toBe(2);
    expect(pageOfPosition(-3)).toBe(1);
  });
});

describe("reasonText", () => {
  it("viết ngắn theo tác vụ không chắc", () => {
    const sentiment = makeRow({ source: "flag", reasons: [lowConfidence("sentiment", "Cảm xúc không chắc — A 40% và B 30%")] });
    const topic = makeRow({ source: "flag", reasons: [lowConfidence("topic", "Chủ đề không chắc — A 40% và B 30%")] });
    expect(reasonText(sentiment, 25)).toBe("Cảm xúc không chắc");
    expect(reasonText(topic, 25)).toBe("Chủ đề không chắc");
  });

  it("cả hai tác vụ không chắc thì ghi gộp, không lặp", () => {
    const row = makeRow({
      source: "flag",
      reasons: [
        lowConfidence("sentiment", "Cảm xúc không chắc"),
        lowConfidence("topic", "Chủ đề không chắc"),
        lowConfidence("topic", "Chủ đề không chắc"),
      ],
    });
    expect(reasonText(row, 25)).toBe("Cả hai không chắc");
  });

  it("câu kiểm tra ngẫu nhiên ghi rõ là mẫu kiểm tra", () => {
    expect(reasonText(makeRow({ source: "audit" }), 25)).toBe("Ngẫu nhiên (kiểm tra)");
  });

  it("câu không dấu ghi riêng, có thể đi cùng lý do không chắc", () => {
    const only = makeRow({
      source: "flag",
      no_diacritics: true,
      reasons: [{ code: "no_diacritics", task: null, candidates: [], message: "Câu không dấu" }],
    });
    expect(reasonText(only, 25)).toBe("Câu không dấu");
    const both = makeRow({
      source: "flag",
      no_diacritics: true,
      reasons: [
        lowConfidence("topic", "Chủ đề không chắc"),
        { code: "no_diacritics", task: null, candidates: [], message: "Câu không dấu" },
      ],
    });
    expect(reasonText(both, 25)).toBe("Chủ đề không chắc · Câu không dấu");
  });

  it("câu dài hơn giới hạn giải thích thì thêm số từ", () => {
    const row = makeRow({
      source: "flag",
      n_words: 44,
      explainable: false,
      reasons: [lowConfidence("topic", "Chủ đề không chắc")],
    });
    expect(reasonText(row, 40)).toBe("Chủ đề không chắc · dài 44 từ");
    expect(reasonText({ ...row, n_words: 40 }, 40)).toBe("Chủ đề không chắc");
  });

  it("dòng ngoài hàng đợi không có lý do thì hiện dấu gạch", () => {
    expect(reasonText(makeRow(), 25)).toBe("—");
  });

  it("dòng bị cờ mà thiếu lý do vẫn có chữ", () => {
    expect(reasonText(makeRow({ source: "flag" }), 25)).toBe("Cần xem xét");
  });
});

describe("reasonDetail", () => {
  it("ghép thông điệp đầy đủ của máy chủ, bỏ trùng", () => {
    const row = makeRow({
      source: "flag",
      reasons: [
        lowConfidence("sentiment", "Cảm xúc không chắc — Tích cực 39% và Trung tính 30%"),
        lowConfidence("sentiment", "Cảm xúc không chắc — Tích cực 39% và Trung tính 30%"),
        lowConfidence("topic", "Chủ đề không chắc — Khác 39% và Giảng viên 37%"),
      ],
    });
    expect(reasonDetail(row)).toBe(
      "Cảm xúc không chắc — Tích cực 39% và Trung tính 30%; Chủ đề không chắc — Khác 39% và Giảng viên 37%",
    );
  });

  it("không có lý do thì không có gợi ý", () => {
    expect(reasonDetail(makeRow())).toBeUndefined();
  });
});
