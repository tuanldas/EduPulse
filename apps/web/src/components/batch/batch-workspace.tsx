"use client";

import { Callout } from "@/components/shared/callout";
import { FailureNotice } from "@/components/shared/failure-notice";
import { Icon } from "@/components/shared/icon";
import { SectionHeading } from "@/components/shared/section-heading";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { retryAppConfig } from "@/lib/app-config";
import { toFailure } from "@/lib/api/client";
import { fileStem } from "@/lib/batch/download";

import { ExportStep } from "./export-step";
import { ResultsSection } from "./results-section";
import { RunProgress } from "./run-progress";
import { SettingsStep } from "./settings-step";
import { UploadStep } from "./upload-step";
import { useBatchFlow } from "./use-batch-flow";

function ResultsSkeleton() {
  return (
    <div className="flex flex-col gap-4" aria-busy="true" aria-label="Đang tải kết quả">
      <div className="grid grid-cols-4 gap-4 max-[900px]:grid-cols-2 max-[640px]:grid-cols-1">
        {[0, 1, 2, 3].map((item) => (
          <Skeleton key={item} className="h-24" />
        ))}
      </div>
      <Skeleton className="h-10 w-72" />
      <Skeleton className="h-72 w-full" />
    </div>
  );
}

/** Trang 2: tải file → chọn cột và ngân sách → chạy nền → duyệt, sửa nhãn → xuất CSV. */
export function BatchWorkspace() {
  const flow = useBatchFlow();
  const { stored, phase, configState, actions } = flow;
  const config = configState.status === "ready" ? configState.config : undefined;
  const results = flow.results.status === "ready" ? flow.results.data : null;
  const showSteps = stored !== null && phase !== "ended";
  const locked = phase === "running" || phase === "checking" || phase === "done" || phase === "unreachable";

  return (
    <>
      <Callout tone="info" icon="lock" title="Dữ liệu chỉ nằm trong phiên làm việc này">
        và mất khi đóng tab. EduPulse tự ẩn SĐT, email, MSSV, CCCD, URL trước khi phân tích;{" "}
        <b className="font-semibold">tên người chỉ ẩn ở mức cố gắng tối đa (best effort)</b>, nên hãy ẩn danh file
        trước khi tải lên.
      </Callout>

      {configState.status === "error" ? (
        <FailureNotice failure={configState.failure} onRetry={() => void retryAppConfig()} />
      ) : null}

      <SectionHeading>1 · Tải file</SectionHeading>
      <UploadStep
        config={config}
        batch={phase === "ended" ? null : stored}
        preview={flow.preview}
        activity={flow.activity}
        failure={flow.uploadFailure}
        locked={phase === "running" || phase === "checking"}
        onFile={actions.uploadFile}
        onSample={actions.loadSample}
        onRemove={actions.removeFile}
      />

      {phase === "ended" ? (
        <Callout
          tone="warn"
          icon="history"
          title="Phiên làm việc đã kết thúc."
          role="alert"
          action={
            <Button variant="outline" size="sm" onClick={actions.reset}>
              <Icon name="restart_alt" size="sm" />
              Bắt đầu lại
            </Button>
          }
        >
          File và kết quả đã bị xoá khỏi máy chủ (hết thời gian chờ hoặc tab trước đã đóng). Hãy tải file lên lại.
        </Callout>
      ) : null}

      {showSteps && config ? (
        <>
          <SectionHeading>2 · Thiết lập</SectionHeading>
          <SettingsStep
            key={stored.batchId}
            batch={stored}
            config={config}
            locked={locked}
            finished={phase === "done"}
            starting={flow.starting}
            failure={flow.runFailure}
            onRun={actions.run}
            onReset={actions.reset}
          />
          {flow.runPhase !== null ? (
            <RunProgress
              phase={flow.runPhase}
              status={flow.status}
              budget={stored.run?.budget ?? config.default_budget}
              autoXaiTop={config.limits.auto_xai_top}
              nRows={stored.nRows}
              elapsedS={results?.summary.elapsed_s ?? null}
              onRetry={actions.retryRun}
            />
          ) : null}
          {phase === "unreachable" && flow.pollFailure !== null ? (
            <FailureNotice failure={flow.pollFailure} onRetry={actions.retryPolling} />
          ) : null}
        </>
      ) : null}

      {showSteps && !config && configState.status === "loading" ? <Skeleton className="h-40 w-full" /> : null}

      {phase === "done" ? (
        <>
          <SectionHeading>3 · Kết quả &amp; duyệt</SectionHeading>
          {flow.results.status === "loading" ? <ResultsSkeleton /> : null}
          {flow.results.status === "error" ? (
            <FailureNotice failure={toFailure(flow.results.failure)} />
          ) : null}
          {results !== null && stored !== null && config ? (
            <>
              <ResultsSection
                key={stored.batchId}
                batchId={stored.batchId}
                results={results}
                groupLabel={stored.run?.groupCol ?? (results.summary.n_groups !== null ? "Nhóm" : null)}
                xaiMaxWords={config.limits.xai_max_words}
                overrides={flow.edits.edits.overrides}
                correctedCount={flow.edits.count}
                onChange={flow.edits.change}
              />
              <SectionHeading>4 · Xuất kết quả</SectionHeading>
              <ExportStep
                batchId={stored.batchId}
                fileStem={fileStem(stored.filename)}
                summary={results.summary}
                overrides={flow.edits.edits.overrides}
                correctedCount={flow.edits.count}
                unexported={flow.edits.unexported}
                onExported={flow.edits.markExported}
              />
            </>
          ) : null}
        </>
      ) : null}
    </>
  );
}
