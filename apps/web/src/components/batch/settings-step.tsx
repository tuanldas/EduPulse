"use client";

import { useId, useState } from "react";

import { WrappingIllustrativeBadge } from "@/components/batch/wrapping-illustrative-badge";
import { Callout } from "@/components/shared/callout";
import { Icon } from "@/components/shared/icon";
import { Caption } from "@/components/shared/section-heading";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import type { ApiFailure } from "@/lib/api/errors";
import type { AppConfig } from "@/lib/api/types";
import { batchFailureMessage } from "@/lib/batch/batch-errors";
import { budgetTarget, sortedBudgets } from "@/lib/batch/budget";
import type { StoredBatch, StoredRun } from "@/lib/batch/stored-batch";
import { formatNumber, formatPercent } from "@/lib/format";
import { cn } from "@/lib/utils";

import { BudgetEfficiency } from "./budget-efficiency";
import { HelpTip } from "./help-tip";
import { NativeSelect } from "./native-select";

interface SettingsStepProps {
  batch: StoredBatch;
  config: AppConfig;
  /** Đang chạy hoặc đã chạy xong: khoá các lựa chọn để kết quả luôn khớp với thiết lập đã dùng. */
  locked: boolean;
  /** Đã chạy xong: máy chủ đã giải phóng dữ liệu thô nên muốn đổi thiết lập phải tải file lại. */
  finished: boolean;
  /** Đang gửi yêu cầu chạy. */
  starting: boolean;
  failure: ApiFailure | null;
  onRun: (run: StoredRun) => void;
  /** Bỏ file hiện tại để bắt đầu lại từ bước 1. */
  onReset: () => void;
}

/** Bước 2: chọn cột văn bản, cột nhóm, ngân sách duyệt rồi chạy. Đổi file thì dựng lại bằng `key`. */
export function SettingsStep({ batch, config, locked, finished, starting, failure, onRun, onReset }: SettingsStepProps) {
  const budgets = sortedBudgets(config.budgets);
  const [textCol, setTextCol] = useState(batch.run?.textCol ?? batch.suggestedTextCol ?? "");
  const [groupCol, setGroupCol] = useState(batch.run?.groupCol ?? batch.suggestedGroupCol ?? "");
  const [budget, setBudget] = useState(batch.run?.budget ?? config.default_budget);
  const textId = useId();
  const groupId = useId();
  const budgetLabelId = useId();

  const budgetIndex = Math.max(
    0,
    budgets.findIndex((option) => option.budget === budget),
  );
  const columnOptions = batch.columns.map((column) => ({ value: column, label: column || "(không tên)" }));
  const groupOptions = columnOptions.filter((option) => option.value !== textCol);
  const overLimit = batch.nRows > config.limits.max_rows;
  const canRun = textCol !== "" && !locked && !starting && !overLimit;
  const target = budgetTarget(budget, batch.nRows);

  function changeTextColumn(next: string) {
    setTextCol(next);
    if (next === groupCol) setGroupCol("");
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 items-end gap-4 @max-[620px]:grid-cols-1">
        <div>
          <div className="mb-1.5 flex items-center gap-1.5 text-sm">
            <label htmlFor={textId}>Cột văn bản</label>
            <HelpTip label="Giải thích cột văn bản">Cột chứa câu góp ý của sinh viên cần phân tích.</HelpTip>
          </div>
          <NativeSelect
            id={textId}
            value={textCol}
            onValueChange={changeTextColumn}
            options={columnOptions}
            placeholder="Chọn cột…"
            disabled={locked}
          />
        </div>
        <div>
          <div className="mb-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-sm">
            <label htmlFor={groupId}>Cột nhóm (tuỳ chọn)</label>
            {batch.illustrativeGroups ? <WrappingIllustrativeBadge /> : null}
          </div>
          <NativeSelect
            id={groupId}
            value={groupCol}
            onValueChange={setGroupCol}
            options={groupOptions}
            placeholder="Không dùng cột nhóm"
            disabled={locked}
          />
        </div>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,2fr)] items-start gap-4 @max-[800px]:grid-cols-1">
        <div className="flex flex-col gap-2">
          <div>
            <div className="mb-1 flex items-center justify-between gap-2 text-sm">
              <span id={budgetLabelId} className="inline-flex items-center gap-1.5">
                Ngân sách duyệt
                <HelpTip label="Giải thích ngân sách duyệt">
                  Tỉ lệ câu có độ tin cậy thấp nhất được chuyển cho người duyệt. Ngân sách lớn hơn bắt được nhiều lỗi
                  hơn nhưng tốn công duyệt hơn.
                </HelpTip>
              </span>
              <b className="font-semibold text-primary-text tabular-nums">{formatPercent(budget / 100)}</b>
            </div>
            <div className="px-2 py-3">
              <Slider
                min={0}
                max={budgets.length - 1}
                step={1}
                value={[budgetIndex]}
                disabled={locked}
                aria-labelledby={budgetLabelId}
                onValueChange={(value) => {
                  const index = Array.isArray(value) ? value[0] : value;
                  const next = budgets[index];
                  if (next !== undefined) setBudget(next.budget);
                }}
              />
            </div>
            <div className="flex justify-between px-2 text-[13px] text-muted-foreground tabular-nums" aria-hidden="true">
              {budgets.map((option) => (
                <span key={option.budget} className={cn(option.budget === budget && "font-semibold text-primary-text")}>
                  {formatPercent(option.budget / 100)}
                </span>
              ))}
            </div>
          </div>
          <Caption>
            <b className="font-semibold text-foreground">
              {formatPercent(budget / 100)} ≈ {formatNumber(target)} / {formatNumber(batch.nRows)} câu
            </b>{" "}
            được gắn cờ “Cần xem xét”: những câu có độ tin cậy thấp nhất ở cảm xúc <b className="font-semibold">hoặc</b>{" "}
            chủ đề; lý do ghi theo tác vụ không chắc.
          </Caption>
        </div>
        <BudgetEfficiency options={budgets} selected={budget} />
      </div>

      {finished ? (
        <Callout
          tone="info"
          icon="info"
          title="Đã phân tích xong."
          role="status"
          action={
            <Button variant="outline" size="sm" onClick={onReset}>
              <Icon name="restart_alt" size="sm" />
              Bắt đầu lại
            </Button>
          }
        >
          Muốn đổi cột văn bản, cột nhóm hoặc ngân sách duyệt, hãy tải file lên lại.
        </Callout>
      ) : (
        <div className="flex flex-wrap items-center gap-4">
          <Button
            disabled={!canRun}
            onClick={() => onRun({ textCol, groupCol: groupCol === "" ? null : groupCol, budget })}
          >
            <Icon name="play_arrow" size="sm" />
            {starting
              ? "Đang gửi…"
              : locked
                ? "Đang phân tích…"
                : `Phân tích ${formatNumber(batch.nRows)} câu`}
          </Button>
          <Caption className="inline-flex items-center gap-1.5">
            <Icon name="schedule" size="xs" />
            Ước tính dưới 2 phút cho 1.000 câu; thanh tiến trình sẽ báo thời gian còn lại.
          </Caption>
        </div>
      )}

      {overLimit ? (
        <Callout tone="warn" icon="content_cut" role="alert">
          File có {formatNumber(batch.nRows)} câu, vượt giới hạn {formatNumber(config.limits.max_rows)} câu mỗi lần. Hãy
          tách file rồi phân tích lần lượt.
        </Callout>
      ) : null}
      {failure ? (
        <Callout tone="err" role="alert">
          {batchFailureMessage(failure, { maxRows: config.limits.max_rows })}
        </Callout>
      ) : null}
    </div>
  );
}
