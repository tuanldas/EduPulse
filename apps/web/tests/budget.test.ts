import { describe, expect, it } from "vitest";

import { acceptGainPoints, budgetTarget, optionFor, sortedBudgets } from "@/lib/batch/budget";
import type { BudgetOption } from "@/lib/batch/types";

function option(budget: BudgetOption["budget"], accept: number): BudgetOption {
  return {
    budget,
    default: budget === 10,
    efficiency: {
      flagged_share: budget / 100,
      caught_sentiment_errors: 0.5,
      caught_topic_errors: 0.4,
      accept_accuracy_both: accept,
      all_accuracy_both: 0.865,
    },
  };
}

describe("ngân sách duyệt", () => {
  const budgets = [option(30, 0.95), option(5, 0.885), option(20, 0.937), option(10, 0.907)];

  it("sắp tăng dần để khớp thứ tự điểm dừng của thanh trượt", () => {
    expect(sortedBudgets(budgets).map((item) => item.budget)).toEqual([5, 10, 20, 30]);
    expect(budgets.map((item) => item.budget)).toEqual([30, 5, 20, 10]);
  });

  it("tìm mức theo giá trị", () => {
    expect(optionFor(budgets, 20)?.efficiency.accept_accuracy_both).toBe(0.937);
    expect(optionFor([], 20)).toBeUndefined();
  });

  it("số câu dự kiến bị cờ là phần nguyên làm tròn lên", () => {
    expect(budgetTarget(10, 1000)).toBe(100);
    expect(budgetTarget(10, 200)).toBe(20);
    expect(budgetTarget(5, 99)).toBe(5);
    expect(budgetTarget(30, 7)).toBe(3);
    expect(budgetTarget(10, 0)).toBe(0);
  });

  it("chênh lệch độ chính xác của phần không cờ so với chung, tính bằng điểm", () => {
    expect(acceptGainPoints(option(10, 0.907).efficiency)).toBeCloseTo(4.2, 6);
  });
});
