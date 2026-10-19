import { Callout } from "@/components/shared/callout";
import { Icon } from "@/components/shared/icon";
import { Button } from "@/components/ui/button";
import { isRetryable, type ApiFailure } from "@/lib/api/errors";
import { batchFailureMessage, type BatchErrorContext } from "@/lib/batch/batch-errors";

interface BatchFailureNoticeProps {
  failure: ApiFailure;
  context?: BatchErrorContext;
  /** Có thì hiện nút "Thử lại" khi lỗi có thể qua đi (mất mạng, máy chủ bận). */
  onRetry?: () => void;
  className?: string;
}

/** Như `FailureNotice` của nền tảng nhưng dùng câu riêng cho các mã lỗi của phân tích hàng loạt và dashboard. */
export function BatchFailureNotice({ failure, context, onRetry, className }: BatchFailureNoticeProps) {
  const retry =
    onRetry && isRetryable(failure) ? (
      <Button variant="outline" size="sm" onClick={onRetry}>
        <Icon name="refresh" size="sm" />
        Thử lại
      </Button>
    ) : undefined;
  return (
    <Callout tone="err" role="alert" action={retry} className={className}>
      {batchFailureMessage(failure, context)}
    </Callout>
  );
}
