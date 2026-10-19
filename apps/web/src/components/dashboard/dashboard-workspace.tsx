"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { Callout } from "@/components/shared/callout";
import { DemoUnavailableNotice } from "@/components/shared/demo-unavailable-notice";
import { FailureNotice } from "@/components/shared/failure-notice";
import { Icon } from "@/components/shared/icon";
import { BatchFailureNotice } from "@/components/batch/batch-failure-notice";
import { Caption } from "@/components/shared/section-heading";
import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { toFailure } from "@/lib/api/client";
import { retryAppConfig, useAppConfig } from "@/lib/app-config";
import { getInsights, sendClose, sendHeartbeat } from "@/lib/batch/batch-api";
import { useReviewEdits } from "@/lib/batch/review-edits";
import { useStoredBatch } from "@/lib/batch/stored-batch";
import { DEMO_BATCH_ID } from "@/lib/batch/types";
import { useAsyncData } from "@/lib/batch/use-async-data";
import { useBatchSession } from "@/lib/batch/use-batch-session";
import { cn } from "@/lib/utils";

import { GroupFilter } from "./group-filter";
import { InsightsView } from "./insights-view";
import { SourceSwitch, type InsightSource } from "./source-switch";

/** Chu kỳ heartbeat dự phòng khi chưa nạp được `/api/config` (khớp `HEARTBEAT_S` của API). */
const FALLBACK_HEARTBEAT_S = 60;

export function DashboardSkeleton() {
  return (
    <div className="@container flex flex-col gap-4" aria-busy="true" aria-label="Đang tải dashboard">
      <Skeleton className="h-16 w-full max-w-xl" />
      <div className="grid grid-cols-4 gap-4 @max-[800px]:grid-cols-2 @max-[620px]:grid-cols-1">
        {[0, 1, 2, 3].map((item) => (
          <Skeleton key={item} className="h-28" />
        ))}
      </div>
      <Skeleton className="h-48 w-full" />
      <div className="grid grid-cols-2 gap-4 @max-[800px]:grid-cols-1">
        <Skeleton className="h-72" />
        <Skeleton className="h-72" />
      </div>
    </div>
  );
}

/**
 * Trang 3: chọn nguồn (dữ liệu demo hoặc file vừa tải), tuỳ chọn lọc nhóm, rồi hiện dashboard. Nguồn lấy
 * từ `?batch=` (có thì là file, không thì là demo nếu có); chuyển trang bằng `<Link>` nên batch không bị
 * đóng khi sang đây, còn heartbeat/close tiếp tục chạy ở trang này khi xem file.
 */
export function DashboardWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const appConfig = useAppConfig();
  const config = appConfig.status === "ready" ? appConfig.config : undefined;
  const stored = useStoredBatch();

  const batchParam = searchParams.get("batch");
  const demoAvailable = config?.demo_available === true;
  const fileId = batchParam !== null && batchParam !== DEMO_BATCH_ID ? batchParam : (stored?.batchId ?? null);

  // `null`: chưa biết (đang nạp cấu hình); "none": không có nguồn nào để xem.
  let source: InsightSource | "none" | null;
  if (batchParam === DEMO_BATCH_ID) source = "demo";
  else if (batchParam !== null) source = "batch";
  else if (config === undefined) {
    // Cấu hình lỗi vẫn xem được file đang có (cấu hình chỉ cho biết demo và chu kỳ heartbeat).
    source = appConfig.status === "error" ? (fileId !== null ? "batch" : "none") : null;
  }
  else if (demoAvailable) source = "demo";
  else if (fileId !== null) source = "batch";
  else source = "none";

  const batchId = source === "demo" ? DEMO_BATCH_ID : source === "batch" ? fileId : null;
  const isFile = source === "batch" && fileId !== null;
  const knownFile = stored !== null && stored.batchId === fileId ? stored : null;

  // --- Bộ lọc nhóm: gắn với batch nên đổi nguồn là tự về "tất cả" ------------------------------
  const [groupChoice, setGroupChoice] = useState<{ batchId: string; groups: string[] } | null>(null);
  const selectedGroups = batchId !== null && groupChoice?.batchId === batchId ? groupChoice.groups : [];

  const insights = useAsyncData(
    batchId === null ? null : `${batchId}|${selectedGroups.join("\u0001")}`,
    (signal) => getInsights(batchId ?? DEMO_BATCH_ID, selectedGroups, signal),
  );
  const data =
    insights.status === "ready" ? insights.data : insights.status === "loading" ? insights.stale : null;
  const refreshing = insights.status === "loading" && data !== null;

  // --- Giữ phiên của file đang xem ------------------------------------------------------------
  const edits = useReviewEdits(isFile ? fileId : null);
  const [endedFor, setEndedFor] = useState<string | null>(null);
  const failure = insights.status === "error" ? toFailure(insights.failure) : null;
  const ended = isFile && (endedFor === fileId || failure?.status === 404);
  useBatchSession({
    batchId: isFile && !ended ? fileId : null,
    heartbeatS: config?.heartbeat_s ?? FALLBACK_HEARTBEAT_S,
    sendHeartbeat,
    sendClose,
    onGone: () => setEndedFor(fileId),
    confirmLeave: edits.unexported,
  });

  function chooseSource(next: InsightSource) {
    if (next === "demo") router.replace("/dashboard/", { scroll: false });
    else if (fileId !== null) router.replace(`/dashboard/?batch=${encodeURIComponent(fileId)}`, { scroll: false });
  }

  const groupLabel = knownFile?.run?.groupCol ?? "nhóm";

  return (
    // Khung đo bề rộng thật của nội dung (đã trừ thanh bên): các lưới bên dưới đổi cột theo đó, không theo bề rộng cửa sổ.
    <div className="@container flex flex-col gap-4">
      <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,1fr)] items-end gap-4 @max-[800px]:grid-cols-1">
        <SourceSwitch
          value={source === "batch" ? "batch" : "demo"}
          onValueChange={chooseSource}
          demoAvailable={demoAvailable}
          demoRows={source === "demo" && selectedGroups.length === 0 && data !== null ? data.kpis.n_rows : null}
          fileAvailable={fileId !== null}
          fileRows={knownFile?.nRows ?? null}
        />
        {data !== null && data.groups !== null && !ended ? (
          <GroupFilter
            columnLabel={groupLabel}
            groups={data.groups}
            selected={selectedGroups}
            illustrative={data.illustrative_groups}
            onChange={(groups) => batchId !== null && setGroupChoice({ batchId, groups })}
          />
        ) : null}
      </div>
      <Caption>
        <Icon name="info" size="xs" /> Mặc định là dữ liệu demo (tập test UIT-VSFC đã dự đoán sẵn). Bộ lọc chỉ hiện khi
        file có cột nhóm. Tỉ lệ tính trên nhãn dự đoán, không phản ánh nhãn bạn đã sửa ở bảng duyệt.
      </Caption>

      {appConfig.status === "error" ? (
        <FailureNotice failure={appConfig.failure} onRetry={() => void retryAppConfig()} />
      ) : null}
      {source === "none" ? (
        <NoSource demoReason={config?.demo_reason} showDemoNotice={config !== undefined && !demoAvailable} />
      ) : null}
      {source === null || (insights.status === "loading" && data === null) ? <DashboardSkeleton /> : null}

      {ended ? (
        <Callout
          tone="warn"
          icon="history"
          title="Phiên làm việc đã kết thúc."
          role="alert"
          action={
            demoAvailable ? (
              <Button variant="outline" size="sm" onClick={() => router.replace("/dashboard/", { scroll: false })}>
                <Icon name="dataset" size="sm" />
                Xem dữ liệu demo
              </Button>
            ) : undefined
          }
        >
          File này không còn trên máy chủ (hết thời gian chờ hoặc tab trước đã đóng). Hãy{" "}
          <Link href="/batch/">phân tích lại file</Link> để xem dashboard.
        </Callout>
      ) : null}

      {failure !== null && !ended ? (
        failure.code === "not_done" ? (
          <Callout tone="info" icon="schedule" title="Phân tích chưa xong." role="status">
            Hãy chờ thanh tiến trình ở trang <Link href="/batch/">Phân tích hàng loạt</Link> hoàn tất rồi quay lại.
          </Callout>
        ) : (
          <BatchFailureNotice failure={failure} onRetry={insights.reload} />
        )
      ) : null}

      {data !== null && !ended ? (
        <div aria-busy={refreshing} className={cn("transition-opacity", refreshing && "opacity-60")}>
          <InsightsView key={batchId} data={data} groupLabel={groupLabel} />
        </div>
      ) : null}
    </div>
  );
}

function NoSource({ demoReason, showDemoNotice }: { demoReason: string | null | undefined; showDemoNotice: boolean }) {
  return (
    <div className="flex flex-col gap-3">
      {showDemoNotice ? <DemoUnavailableNotice reason={demoReason} /> : null}
      <Callout
        tone="info"
        icon="upload_file"
        title="Chưa có dữ liệu để xem."
        role="status"
        action={
          <Link href="/batch/" className={cn(buttonVariants({ variant: "outline", size: "sm" }))}>
            Phân tích một file
            <Icon name="arrow_forward" size="sm" />
          </Link>
        }
      >
        Hãy phân tích một file ở trang Phân tích hàng loạt, rồi quay lại đây để xem dashboard của file đó.
      </Callout>
    </div>
  );
}
