"use client";

import { useId, useState } from "react";

import { Callout } from "@/components/shared/callout";
import { DemoUnavailableNotice } from "@/components/shared/demo-unavailable-notice";
import { Icon } from "@/components/shared/icon";
import { Caption } from "@/components/shared/section-heading";
import { Button, buttonVariants } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { ApiFailure } from "@/lib/api/errors";
import type { AppConfig } from "@/lib/api/types";
import { batchFailureMessage } from "@/lib/batch/batch-errors";
import type { StoredBatch } from "@/lib/batch/stored-batch";
import type { BatchWarning } from "@/lib/batch/types";
import { formatBytes, formatNumber } from "@/lib/format";
import { cn } from "@/lib/utils";

import { HelpTip } from "./help-tip";

/** Số câu của file mẫu, khớp `SAMPLE_SIZE` ở `edupulse.demo_data`; chỉ để ghi trên nút. */
const SAMPLE_ROWS = 200;

export type UploadActivity = "idle" | "uploading" | "sampling";

interface DropzoneProps {
  maxUploadMb: number | undefined;
  disabled: boolean;
  activity: UploadActivity;
  onFile: (file: File) => void;
}

function Dropzone({ maxUploadMb, disabled, activity, onFile }: DropzoneProps) {
  const inputId = useId();
  const [dragging, setDragging] = useState(false);
  const busy = activity === "uploading";

  return (
    <label
      htmlFor={inputId}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const file = event.dataTransfer.files[0];
        if (!disabled && file) onFile(file);
      }}
      className={cn(
        "flex cursor-pointer flex-wrap items-center gap-4 rounded-lg border border-border bg-muted p-4 transition-colors has-[:focus-visible]:ring-3 has-[:focus-visible]:ring-ring/50",
        dragging && "border-primary bg-primary-soft",
        disabled && "cursor-not-allowed opacity-60",
      )}
    >
      {busy ? (
        <span
          aria-hidden="true"
          className="size-9 shrink-0 animate-spin rounded-full border-[3px] border-grid border-t-primary motion-reduce:animate-none"
        />
      ) : (
        <Icon name="cloud_upload" className="size-9 text-muted-foreground" />
      )}
      <div className="min-w-[150px] flex-1">
        <div className="text-[15px]">{busy ? "Đang tải file lên…" : "Kéo thả file vào đây"}</div>
        <div className="text-[13px] text-muted-foreground">
          {maxUploadMb ? `Giới hạn ${formatNumber(maxUploadMb)} MB mỗi file • ` : ""}CSV, XLSX
        </div>
      </div>
      <span className={cn(buttonVariants({ variant: "outline", size: "sm" }), "pointer-events-none")}>Chọn file</span>
      <input
        id={inputId}
        type="file"
        accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        className="sr-only"
        disabled={disabled}
        onChange={(event) => {
          const file = event.target.files?.[0];
          // Cho phép chọn lại đúng file đó lần nữa (sau khi sửa file hoặc bị lỗi).
          event.target.value = "";
          if (file) onFile(file);
        }}
      />
    </label>
  );
}

function WarningNotice({ warning }: { warning: BatchWarning }) {
  const count = warning.count === null ? null : formatNumber(warning.count);
  switch (warning.code) {
    case "encoding_fallback":
      return (
        <Callout tone="warn" icon="warning" title="File không phải UTF-8." role="status">
          Đã đọc theo bảng mã Windows-1258, dấu tiếng Việt có thể sai. Nếu thấy chữ lạ, hãy lưu lại file bằng UTF-8.
        </Callout>
      );
    case "blank_rows_skipped":
      return (
        <Callout tone="info" role="status">
          Đã bỏ {count ?? "các"} dòng trống.
        </Callout>
      );
    case "other_sheets_ignored":
      return (
        <Callout tone="info" role="status">
          File có nhiều sheet; chỉ đọc sheet đầu tiên.
        </Callout>
      );
    case "columns_renamed":
      return (
        <Callout tone="info" role="status">
          {count ?? "Một số"} cột có tiêu đề trống hoặc trùng nên đã được đặt lại tên.
        </Callout>
      );
    default:
      return (
        <Callout tone="info" role="status">
          {warning.message}
        </Callout>
      );
  }
}

interface PreviewProps {
  columns: readonly string[];
  rows: readonly (readonly string[])[];
  nRows: number;
}

function Preview({ columns, rows, nRows }: PreviewProps) {
  return (
    <Collapsible className="rounded-lg border border-border">
      <CollapsibleTrigger className="group flex min-h-11 w-full items-center gap-2.5 rounded-lg px-3.5 text-left text-[15px] outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
        <Icon name="table_view" className="text-muted-foreground" />
        <span>
          Xem trước {formatNumber(rows.length)} dòng đầu · {formatNumber(nRows)} dòng × {formatNumber(columns.length)} cột
        </span>
        <Icon
          name="keyboard_arrow_down"
          className="ml-auto text-muted-foreground transition-transform group-data-[panel-open]:rotate-180"
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="flex flex-col gap-2 border-t border-divider px-4 pt-2 pb-4">
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((column, index) => (
                <TableHead key={`${column}-${index}`}>{column || "(không tên)"}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, rowIndex) => (
              <TableRow key={rowIndex}>
                {columns.map((_, columnIndex) => (
                  <TableCell key={columnIndex} className="max-w-[280px] truncate" title={row[columnIndex]}>
                    {row[columnIndex]}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <Caption>
          Phần xem trước hiện nguyên văn file của bạn (chưa ẩn thông tin cá nhân) và chỉ tồn tại trong tab này.
        </Caption>
      </CollapsibleContent>
    </Collapsible>
  );
}

export interface UploadStepProps {
  config: AppConfig | undefined;
  batch: StoredBatch | null;
  /** Xem trước các dòng đầu; chỉ có ngay sau khi tải lên (không lưu qua các lần chuyển trang). */
  preview: readonly (readonly string[])[] | null;
  activity: UploadActivity;
  /** Lỗi của lần tải file hoặc tạo file mẫu gần nhất. */
  failure: ApiFailure | null;
  /** Đang chạy phân tích: chặn đổi file. */
  locked: boolean;
  onFile: (file: File) => void;
  onSample: () => void;
  onRemove: () => void;
}

/** Bước 1: tải file (kéo thả hoặc chọn), hoặc dùng file mẫu; hiện file đang dùng, cảnh báo và xem trước. */
export function UploadStep({
  config,
  batch,
  preview,
  activity,
  failure,
  locked,
  onFile,
  onSample,
  onRemove,
}: UploadStepProps) {
  const disabled = activity !== "idle" || locked;
  const maxUploadMb = config?.limits.max_upload_mb;
  const maxRows = config?.limits.max_rows;
  const tooManyRows = failure?.code === "too_many_rows";

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,1fr)] items-end gap-4 max-[640px]:grid-cols-1">
        <div>
          <div className="mb-1.5 flex items-center gap-1.5 text-sm">
            <span>
              File góp ý (CSV hoặc XLSX
              {maxUploadMb ? ` · tối đa ${formatNumber(maxUploadMb)} MB` : ""}
              {maxRows ? ` · ${formatNumber(maxRows)} câu mỗi lần` : ""})
            </span>
            <HelpTip label="Yêu cầu về file">
              Cần một dòng tiêu đề và ít nhất một cột chứa câu góp ý. File XLSX chỉ đọc sheet đầu tiên.
            </HelpTip>
          </div>
          <Dropzone maxUploadMb={maxUploadMb} disabled={disabled} activity={activity} onFile={onFile} />
          {batch ? (
            <div className="flex items-center gap-2.5 px-1 pt-2 text-sm">
              <Icon name="description" className="text-muted-foreground" />
              <b className="min-w-0 truncate font-semibold">{batch.filename}</b>
              <span className="shrink-0 text-muted-foreground">
                {batch.sizeBytes === null ? "" : `${formatBytes(batch.sizeBytes)} · `}
                {formatNumber(batch.nRows)} dòng
              </span>
              <span className="flex-1" />
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label={`Bỏ file ${batch.filename}`}
                disabled={disabled}
                onClick={onRemove}
              >
                <Icon name="close" size="xs" />
              </Button>
            </div>
          ) : null}
        </div>

        {config?.demo_available === false ? (
          <DemoUnavailableNotice reason={config.demo_reason} />
        ) : (
          <div className="flex flex-col gap-2">
            <Button variant="outline" className="w-full" disabled={disabled || config === undefined} onClick={onSample}>
              {activity === "sampling" ? (
                <span
                  aria-hidden="true"
                  className="size-4 animate-spin rounded-full border-2 border-grid border-t-primary motion-reduce:animate-none"
                />
              ) : (
                <Icon name="dataset" size="sm" />
              )}
              Dùng file mẫu ({formatNumber(SAMPLE_ROWS)} câu UIT-VSFC)
            </Button>
            <Caption>Dành cho demo khi chưa có file thật.</Caption>
          </div>
        )}
      </div>

      {failure ? (
        tooManyRows ? (
          <div className="flex flex-col gap-3">
            <Callout tone="warn" icon="content_cut" role="alert">
              {batchFailureMessage(failure, { maxUploadMb, maxRows })}
            </Callout>
            <Button disabled className="self-start">
              <Icon name="play_arrow" size="sm" />
              Phân tích
            </Button>
            <Caption>Giới hạn do máy chủ cấu hình để giữ bộ nhớ và thời gian chạy ổn định.</Caption>
          </div>
        ) : (
          <Callout tone="err" role="alert">
            {batchFailureMessage(failure, { maxUploadMb, maxRows })}
          </Callout>
        )
      ) : null}

      {batch?.warnings.map((warning, index) => (
        <WarningNotice key={`${warning.code}-${index}`} warning={warning} />
      ))}

      {batch && preview && preview.length > 0 ? (
        <Preview columns={batch.columns} rows={preview} nRows={batch.nRows} />
      ) : null}
    </div>
  );
}
