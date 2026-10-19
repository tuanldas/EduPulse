"use client";

import { useEffect } from "react";

import { Callout } from "@/components/shared/callout";
import { FailureNotice } from "@/components/shared/failure-notice";
import { Icon, type IconName } from "@/components/shared/icon";
import { Caption, SectionHeading } from "@/components/shared/section-heading";
import { PageHeader } from "@/components/shell/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { anchorId } from "@/lib/anchor";
import type { ApiFailure } from "@/lib/api/errors";

import { LicenseCitation } from "./license-citation";
import { ModelCardSection } from "./model-card";
import { PerfSection } from "./perf-section";
import { PrivacyBlocks } from "./privacy-blocks";
import { useModelCard } from "./use-model-card";

const TOC: readonly { href: string; label: string; icon: IconName }[] = [
  { href: "#hieu-nang", label: "Hiệu năng", icon: "speed" },
  { href: "#model-card", label: "Model card", icon: "badge" },
  { href: "#rieng-tu", label: "Quyền riêng tư", icon: "lock" },
  { href: "#giay-phep", label: "Giấy phép & trích dẫn", icon: "gavel" },
];

/** Giữ đủ mục (và neo của mục lục) khi model card đang tải hoặc không tải được. */
function SectionPlaceholder({
  id,
  icon,
  title,
  children,
}: {
  id: string;
  icon: IconName;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section aria-labelledby={id} className="flex flex-col gap-4">
      <SectionHeading id={id} icon={icon} className="mt-4">
        {title}
      </SectionHeading>
      {children}
    </section>
  );
}

function CardUnavailable({ failure, onRetry }: { failure: ApiFailure; onRetry: () => void }) {
  return failure.code === "model_card_unavailable" ? (
    <Callout tone="info" icon="info" role="status">
      Chưa có model card cho bản mô hình đang chạy, nên chưa hiển thị được số liệu hiệu năng.
    </Callout>
  ) : (
    <FailureNotice failure={failure} onRetry={onRetry} />
  );
}

/** Trang 4: mô hình tốt đến đâu, dùng vào việc gì, giới hạn, quyền riêng tư, giấy phép. Số liệu lấy từ `/api/model-card`. */
export function ModelPage() {
  const { state, retry } = useModelCard();

  // Mở thẳng /model/#giay-phep: lúc trình duyệt cuộn tới neo thì model card chưa về nên trang còn thấp;
  // cuộn lại khi dữ liệu đã nạp (thành công hay lỗi) và bố cục đã ổn định.
  useEffect(() => {
    if (state.status === "loading") return;
    const id = anchorId(window.location.hash);
    if (id) document.getElementById(id)?.scrollIntoView();
  }, [state.status]);

  return (
    <>
      <PageHeader
        icon="balance"
        title="Mô hình & đạo đức"
        description="Mô hình tốt đến đâu, được làm để dùng vào việc gì, và những giới hạn cần biết trước khi tin vào kết quả."
      />

      <Callout tone="warn" icon="gavel" title="AI gợi ý — cần người xem xét.">
        Không dùng kết quả để đánh giá, xếp loại hay kỷ luật cá nhân giảng viên, và không dùng để truy vết sinh viên.
      </Callout>

      <nav aria-label="Mục lục" className="flex flex-wrap gap-x-[18px] gap-y-1.5 text-sm">
        {TOC.map((item) => (
          <a key={item.href} href={item.href} className="inline-flex items-center gap-1 text-link underline-offset-2 hover:underline">
            <Icon name={item.icon} size="xs" />
            {item.label}
          </a>
        ))}
      </nav>

      {state.status === "ready" ? (
        <>
          <PerfSection card={state.card} />
          <ModelCardSection card={state.card} />
        </>
      ) : (
        <>
          <SectionPlaceholder id="hieu-nang" icon="speed" title="Hiệu năng">
            {state.status === "loading" ? (
              <div className="flex flex-col gap-4" aria-busy="true" aria-label="Đang tải model card">
                <div className="grid grid-cols-4 gap-4 max-[1100px]:grid-cols-2 max-[640px]:grid-cols-1">
                  <Skeleton className="h-28" />
                  <Skeleton className="h-28" />
                  <Skeleton className="h-28" />
                  <Skeleton className="h-28" />
                </div>
                <Skeleton className="h-64" />
              </div>
            ) : (
              <CardUnavailable failure={state.failure} onRetry={retry} />
            )}
          </SectionPlaceholder>
          <SectionPlaceholder id="model-card" icon="badge" title="Model card">
            {state.status === "loading" ? <Skeleton className="h-72" /> : null}
          </SectionPlaceholder>
        </>
      )}

      <PrivacyBlocks />
      <LicenseCitation />

      <Caption className="mt-2">EduPulse · đồ án học thuật · 2026.</Caption>
    </>
  );
}
