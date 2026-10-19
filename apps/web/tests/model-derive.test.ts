import { describe, expect, it } from "vitest";

import { caughtErrors, defaultBudgetRow, misclassifiedInto, weakestClass } from "@/components/model/derive";
import type { ModelCard } from "@/lib/api/types";
import { calibrationExample, regroupReliability, trimLeadingEmpty } from "@/lib/reliability";

import { fixture } from "./helpers/api-stub";

const card = fixture<ModelCard>("model-card.json");
const sentiment = card.metrics.sentiment;

describe("gom khoảng độ tin cậy", () => {
  it("bảo toàn tổng số câu và phủ trọn 0..1 bằng các khoảng rộng 0,1", () => {
    const grouped = regroupReliability(sentiment.reliability);
    const before = sentiment.reliability.reduce((sum, bin) => sum + bin.n, 0);
    expect(grouped.reduce((sum, bin) => sum + bin.n, 0)).toBe(before);
    expect(grouped).toHaveLength(10);
    expect(grouped[0].lower).toBe(0);
    expect(grouped[9].upper).toBe(1);
    for (const bin of grouped) {
      if (bin.n === 0) {
        expect(bin.accuracy).toBeNull();
        expect(bin.confidence).toBeNull();
      } else {
        expect(bin.accuracy).toBeGreaterThanOrEqual(0);
        expect(bin.accuracy).toBeLessThanOrEqual(1);
      }
    }
  });

  it("tính trung bình có trọng số theo số câu khi gộp hai khoảng mịn", () => {
    const [group] = regroupReliability(
      [
        { lower: 0.8, upper: 0.85, n: 2, confidence: 0.82, accuracy: 1 },
        { lower: 0.85, upper: 0.9, n: 6, confidence: 0.88, accuracy: 0.5 },
      ],
      0.1,
    ).filter((bin) => bin.n > 0);
    expect(group.lower).toBe(0.8);
    expect(group.n).toBe(8);
    expect(group.accuracy).toBeCloseTo((2 * 1 + 6 * 0.5) / 8, 10);
    expect(group.confidence).toBeCloseTo((2 * 0.82 + 6 * 0.88) / 8, 10);
  });

  it("bỏ các khoảng rỗng ở đầu và để rỗng khi không có dữ liệu", () => {
    const trimmed = trimLeadingEmpty(regroupReliability(sentiment.reliability));
    expect(trimmed[0].n).toBeGreaterThan(0);
    expect(trimLeadingEmpty(regroupReliability([]))).toEqual([]);
  });

  it("ví dụ đọc biểu đồ là khoảng nhiều câu nhất trong các khoảng chưa chạm trần", () => {
    const bins = trimLeadingEmpty(regroupReliability(sentiment.reliability));
    const example = calibrationExample(bins);
    expect(example).toBeDefined();
    expect(example!.upper).toBeLessThanOrEqual(0.9 + 1e-9);
    const rivals = bins.filter((bin) => bin.upper <= 0.9 + 1e-9);
    expect(example!.n).toBe(Math.max(...rivals.map((bin) => bin.n)));
    expect(calibrationExample([])).toBeUndefined();
  });
});

describe("số mô tả suy ra từ model card", () => {
  it("lớp yếu nhất là lớp có F1 thấp nhất và tỉ lệ số câu cộng lại hợp lý", () => {
    const weakest = weakestClass(sentiment)!;
    const lowest = Math.min(...sentiment.per_class.map((item) => item.f1));
    expect(weakest.f1).toBe(lowest);
    const total = sentiment.per_class.reduce((sum, item) => sum + item.support, 0);
    const own = sentiment.per_class.find((item) => item.label === weakest.label)!;
    expect(weakest.share).toBeCloseTo(own.support / total, 10);
  });

  it("không có lớp nào có câu thì không có lớp yếu nhất", () => {
    expect(weakestClass({ ...sentiment, per_class: [] })).toBeUndefined();
  });

  it("câu của lớp bị đoán nhầm sang các lớp khác, nhiều nhất trước, bỏ qua ô nhỏ", () => {
    const weakest = weakestClass(sentiment)!;
    const { into } = misclassifiedInto(sentiment, weakest.label);
    for (const cell of into) {
      expect(cell.label).not.toBe(weakest.label);
      expect(cell.share).toBeGreaterThanOrEqual(0.05);
    }
    expect([...into].sort((a, b) => b.share - a.share)).toEqual(into);
    expect(misclassifiedInto(sentiment, weakest.label, 1.1).into).toEqual([]);
    expect(misclassifiedInto(sentiment, "khong_co").into).toEqual([]);
  });

  it("tìm hàng ngân sách mặc định và số lỗi được bắt theo tác vụ", () => {
    const row = defaultBudgetRow(card)!;
    expect(row.default).toBe(true);
    expect(caughtErrors(row, "sentiment")).toBe(row.efficiency.caught_sentiment_errors);
    expect(caughtErrors(row, "topic")).toBe(row.efficiency.caught_topic_errors);
    expect(defaultBudgetRow({ review_budget: [] })).toBeUndefined();
  });
});
