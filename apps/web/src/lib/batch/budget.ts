import type { BudgetEfficiency, BudgetOption, ReviewBudget } from "./types";

/** Các mức ngân sách theo thứ tự tăng dần: thứ tự của các điểm dừng trên thanh trượt. */
export function sortedBudgets(budgets: readonly BudgetOption[]): BudgetOption[] {
  return [...budgets].sort((a, b) => a.budget - b.budget);
}

export function optionFor(
  budgets: readonly BudgetOption[],
  budget: ReviewBudget,
): BudgetOption | undefined {
  return budgets.find((option) => option.budget === budget);
}

/** Số câu dự kiến bị gắn cờ: ⌈ngân sách · số câu⌉, tính bằng số nguyên để khỏi sai số dấu phẩy động. */
export function budgetTarget(budget: ReviewBudget, nRows: number): number {
  return Math.ceil((budget * nRows) / 100);
}

/** Độ chính xác của phần không bị cờ hơn độ chính xác chung bao nhiêu điểm phần trăm. */
export function acceptGainPoints(efficiency: BudgetEfficiency): number {
  return (efficiency.accept_accuracy_both - efficiency.all_accuracy_both) * 100;
}
