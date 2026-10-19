"use client";

import { Icon } from "@/components/shared/icon";
import { PiiText } from "@/components/shared/pii-chip";
import { Caption } from "@/components/shared/section-heading";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { TONE_CLASS } from "@/lib/colors";
import type { LabelChange, LabelPatch, Overrides } from "@/lib/batch/overrides";
import { paginate, reasonText, REVIEW_PAGE_SIZE } from "@/lib/batch/review-rows";
import type { ResultRow, SentimentLabel, TopicLabel } from "@/lib/batch/types";
import { formatNumber } from "@/lib/format";
import { LABEL_ORDER, labelMeta } from "@/lib/labels";
import { cn } from "@/lib/utils";

import { NativeSelect } from "./native-select";

function isLowConfidence(row: ResultRow, task: "sentiment" | "topic"): boolean {
  return row.reasons.some((reason) => reason.code === "low_confidence" && reason.task === task);
}

function SentimentText({ label }: { label: SentimentLabel }) {
  const meta = labelMeta("sentiment", label);
  return (
    <span className={cn("inline-flex items-center gap-1 font-medium whitespace-nowrap", TONE_CLASS[meta.tone].text)}>
      <Icon name={meta.icon} size="xs" />
      {meta.vi}
    </span>
  );
}

function ConfidenceCell({ value, flagged }: { value: number; flagged: boolean }) {
  const percent = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div className="flex min-w-24 items-center gap-2 tabular-nums">
      <span className="h-1.5 min-w-10 flex-1 overflow-hidden rounded-[3px] bg-grid" aria-hidden="true">
        <i className={cn("block h-full", flagged ? "bg-flag" : "bg-primary")} style={{ width: `${percent}%` }} />
      </span>
      <span className={cn(flagged && "font-semibold text-flag-fg")}>{percent}%</span>
    </div>
  );
}

function editedSummary(patch: LabelPatch | undefined): string {
  if (patch === undefined) return "—";
  const parts: string[] = [];
  if (patch.sentiment !== undefined) parts.push(`Cảm xúc → ${labelMeta("sentiment", patch.sentiment).vi}`);
  if (patch.topic !== undefined) parts.push(`Chủ đề → ${labelMeta("topic", patch.topic).vi}`);
  return parts.join("; ");
}

export interface ReviewTableProps {
  /** Các dòng của tab hiện tại, đã theo thứ tự cần hiển thị (xem `rowsForTab`). */
  rows: readonly ResultRow[];
  overrides: Overrides;
  /** Tab duyệt cho sửa nhãn; tab "Tất cả" chỉ xem và có cột "Đã sửa". */
  editable: boolean;
  /** Tên cột nhóm đã chọn; null nếu không dùng nhóm (ẩn cột). */
  groupLabel: string | null;
  /** `limits.xai_max_words`: câu dài hơn thì cột "Lý do" ghi thêm số từ. */
  xaiMaxWords: number;
  page: number;
  onPageChange: (page: number) => void;
  onChange: (index: number, change: LabelChange) => void;
  /** Mở ngăn giải thích cho dòng ở `position` (vị trí trong `rows`, không phải trong trang). */
  onExplain: (position: number) => void;
}

/**
 * Bảng duyệt: thứ tự dòng cố định (đổi nhãn không sắp lại), 50 dòng mỗi trang. Nhãn gốc của mô hình
 * luôn hiện ở cột "Cảm xúc"/"Chủ đề"; nhãn người duyệt sửa nằm ở cột "Sửa …" riêng (trống = giữ nhãn mô hình).
 */
export function ReviewTable({
  rows,
  overrides,
  editable,
  groupLabel,
  xaiMaxWords,
  page,
  onPageChange,
  onChange,
  onExplain,
}: ReviewTableProps) {
  const view = paginate(rows, page);
  const offset = (view.page - 1) * REVIEW_PAGE_SIZE;
  // # · Câu · Lý do · Cảm xúc · Chủ đề · Giải thích · Tin cậy CX · Tin cậy CĐ, cộng cột sửa/đã sửa và nhóm.
  const columnCount = 8 + (editable ? 2 : 1) + (groupLabel !== null ? 1 : 0);

  return (
    <div className="flex flex-col gap-3">
      <Table className="min-w-[1080px]">
        <caption className="sr-only">
          {editable ? "Danh sách câu cần duyệt, có cột để sửa nhãn" : "Toàn bộ kết quả phân tích"}
        </caption>
        <TableHeader>
          <TableRow>
            <TableHead scope="col" className="w-px text-right">
              #
            </TableHead>
            <TableHead scope="col">Câu (đã ẩn thông tin cá nhân)</TableHead>
            <TableHead scope="col">Lý do</TableHead>
            <TableHead scope="col">Cảm xúc</TableHead>
            {editable ? <TableHead scope="col">Sửa cảm xúc</TableHead> : null}
            <TableHead scope="col">Chủ đề</TableHead>
            {editable ? <TableHead scope="col">Sửa chủ đề</TableHead> : <TableHead scope="col">Đã sửa</TableHead>}
            <TableHead scope="col">Giải thích</TableHead>
            <TableHead scope="col">Tin cậy CX</TableHead>
            <TableHead scope="col">Tin cậy CĐ</TableHead>
            {groupLabel !== null ? <TableHead scope="col">{groupLabel}</TableHead> : null}
          </TableRow>
        </TableHeader>
        <TableBody>
          {view.items.map((row, offsetInPage) => {
            const patch = overrides[row.index];
            return (
              <TableRow key={row.index} data-highlight={patch !== undefined}>
                <TableCell className="text-right text-muted-foreground tabular-nums">{row.source_row}</TableCell>
                <TableCell>
                  <div className="max-w-[260px] truncate" title={row.text}>
                    <PiiText text={row.text} />
                  </div>
                </TableCell>
                <TableCell className={cn("whitespace-nowrap", row.source === "audit" && "text-muted-foreground")}>
                  {reasonText(row, xaiMaxWords)}
                </TableCell>
                <TableCell>
                  <SentimentText label={row.sentiment} />
                </TableCell>
                {editable ? (
                  <TableCell className="min-w-[124px]">
                    <NativeSelect
                      size="sm"
                      aria-label={`Sửa cảm xúc câu ${row.source_row}`}
                      value={patch?.sentiment ?? ""}
                      placeholder="—"
                      options={LABEL_ORDER.sentiment
                        .filter((label) => label !== row.sentiment)
                        .map((label) => ({ value: label, label: labelMeta("sentiment", label).vi }))}
                      onValueChange={(value) =>
                        onChange(row.index, { sentiment: value === "" ? null : (value as SentimentLabel) })
                      }
                    />
                  </TableCell>
                ) : null}
                <TableCell className="whitespace-nowrap">{labelMeta("topic", row.topic).vi}</TableCell>
                {editable ? (
                  <TableCell className="min-w-[148px]">
                    <NativeSelect
                      size="sm"
                      aria-label={`Sửa chủ đề câu ${row.source_row}`}
                      value={patch?.topic ?? ""}
                      placeholder="—"
                      options={LABEL_ORDER.topic
                        .filter((label) => label !== row.topic)
                        .map((label) => ({ value: label, label: labelMeta("topic", label).vi }))}
                      onValueChange={(value) =>
                        onChange(row.index, { topic: value === "" ? null : (value as TopicLabel) })
                      }
                    />
                  </TableCell>
                ) : (
                  <TableCell className="whitespace-nowrap">{editedSummary(patch)}</TableCell>
                )}
                <TableCell>
                  <Button
                    variant="ghost"
                    size="sm"
                    aria-label={`Giải thích câu ${row.source_row}`}
                    onClick={() => onExplain(offset + offsetInPage)}
                  >
                    <Icon name="manage_search" size="sm" />
                    Giải thích
                  </Button>
                </TableCell>
                <TableCell>
                  <ConfidenceCell value={row.sentiment_confidence} flagged={isLowConfidence(row, "sentiment")} />
                </TableCell>
                <TableCell>
                  <ConfidenceCell value={row.topic_confidence} flagged={isLowConfidence(row, "topic")} />
                </TableCell>
                {groupLabel !== null ? (
                  <TableCell className="whitespace-nowrap">{row.group ?? "—"}</TableCell>
                ) : null}
              </TableRow>
            );
          })}
          {view.items.length === 0 ? (
            <TableRow>
              <TableCell colSpan={columnCount} className="py-6 text-center text-muted-foreground">
                Không có câu nào để hiển thị.
              </TableCell>
            </TableRow>
          ) : null}
        </TableBody>
      </Table>

      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 text-sm">
        <span className="text-muted-foreground tabular-nums">
          Hiển thị {formatNumber(view.first)}–{formatNumber(view.last)} / {formatNumber(view.total)} dòng
        </span>
        {view.pageCount > 1 ? (
          <nav aria-label="Phân trang" className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon-sm"
              aria-label="Trang trước"
              disabled={view.page <= 1}
              onClick={() => onPageChange(view.page - 1)}
            >
              <Icon name="chevron_left" size="sm" />
            </Button>
            <span className="min-w-20 text-center tabular-nums" aria-live="polite">
              Trang {formatNumber(view.page)}/{formatNumber(view.pageCount)}
            </span>
            <Button
              variant="outline"
              size="icon-sm"
              aria-label="Trang sau"
              disabled={view.page >= view.pageCount}
              onClick={() => onPageChange(view.page + 1)}
            >
              <Icon name="chevron_right" size="sm" />
            </Button>
          </nav>
        ) : null}
      </div>
      <Caption className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="inline-flex items-center gap-1">
          <Icon name="swipe_left" size="xs" /> Cuộn ngang để xem Tin cậy CX, Tin cậy CĐ{groupLabel !== null ? " và nhóm" : ""}.
        </span>
        <span>Câu dài bị cắt “…”: rê chuột để xem đủ, hoặc bấm “Giải thích”.</span>
      </Caption>
    </div>
  );
}
