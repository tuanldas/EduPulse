"use client";

import Link from "next/link";
import { useState } from "react";
import { toast } from "sonner";

import { Callout } from "@/components/shared/callout";
import { FailureNotice } from "@/components/shared/failure-notice";
import { Icon } from "@/components/shared/icon";
import { Caption } from "@/components/shared/section-heading";
import { Button, buttonVariants } from "@/components/ui/button";
import { toFailure } from "@/lib/api/client";
import { isAbortError } from "@/lib/api/errors";
import { exportBatch } from "@/lib/batch/batch-api";
import { saveBlob } from "@/lib/batch/download";
import { toOverrideList, type Overrides } from "@/lib/batch/overrides";
import type { ResultsSummary } from "@/lib/batch/types";
import { formatNumber } from "@/lib/format";
import { cn } from "@/lib/utils";

type ExportKind = "all" | "corrected";

interface ExportStepProps {
  batchId: string;
  /** Phần tên file đã tải lên (không đuôi), để đặt tên file xuất. */
  fileStem: string;
  summary: ResultsSummary;
  overrides: Overrides;
  /** Số câu có nhãn đã sửa. */
  correctedCount: number;
  /** Có nhãn sửa chưa nằm trong file nào đã tải. */
  unexported: boolean;
  /** Gọi sau khi tạo file thành công, để ghi nhận nhãn sửa đã được xuất. */
  onExported: () => void;
}

/** Bước 4: tải toàn bộ kết quả hoặc chỉ nhãn đã sửa (CSV), sang dashboard, nhắc tải trước khi đóng tab. */
export function ExportStep({
  batchId,
  fileStem,
  summary,
  overrides,
  correctedCount,
  unexported,
  onExported,
}: ExportStepProps) {
  const [busy, setBusy] = useState<ExportKind | null>(null);
  const [failure, setFailure] = useState<ReturnType<typeof toFailure> | null>(null);

  async function download(kind: ExportKind) {
    setBusy(kind);
    setFailure(null);
    try {
      const blob = await exportBatch(batchId, { kind, overrides: toOverrideList(overrides) });
      saveBlob(blob, `${fileStem}_${kind === "all" ? "ket-qua" : "nhan-da-sua"}.csv`);
      onExported();
      const rows = kind === "all" ? summary.n_rows : correctedCount;
      toast.success(`Đã tạo file CSV · ${formatNumber(rows)} dòng`);
    } catch (error) {
      if (!isAbortError(error)) setFailure(toFailure(error));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button disabled={busy !== null} onClick={() => void download("all")}>
          <Icon name="download" size="sm" />
          {busy === "all" ? "Đang tạo file…" : "Tải toàn bộ kết quả (CSV)"}
        </Button>
        <Button
          variant="outline"
          disabled={busy !== null || correctedCount === 0}
          onClick={() => void download("corrected")}
        >
          <Icon name="download" size="sm" />
          {busy === "corrected" ? "Đang tạo file…" : `Tải nhãn đã sửa · ${formatNumber(correctedCount)} dòng`}
        </Button>
        <span className="flex-1" />
        <Link
          href={`/dashboard/?batch=${encodeURIComponent(batchId)}`}
          className={cn(buttonVariants({ variant: "ghost" }))}
        >
          <Icon name="monitoring" size="sm" />
          Xem dashboard cho file này
          <Icon name="arrow_forward" size="sm" />
        </Link>
      </div>
      <Caption>
        CSV mã hoá UTF-8 có BOM (mở thẳng bằng Excel). Chỉ chứa câu đã ẩn thông tin cá nhân, nhãn dự đoán, độ tin cậy,
        nhãn đã sửa và nguồn (cờ / ngẫu nhiên).
      </Caption>
      {failure ? <FailureNotice failure={failure} onRetry={() => void download(busy ?? "all")} /> : null}
      <Callout tone="warn" icon="schedule" title="Kết quả không được lưu trên máy chủ.">
        Hãy tải file trước khi đóng tab hoặc tải file khác.
        {unexported ? (
          <>
            {" "}
            <b className="font-semibold">Còn {formatNumber(correctedCount)} câu có nhãn đã sửa chưa tải xuống.</b>
          </>
        ) : null}
      </Callout>
    </div>
  );
}
