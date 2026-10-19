import { Button } from "@/components/ui/button";
import { failureMessage, isRetryable, type ApiFailure } from "@/lib/api/errors";

import { Callout } from "./callout";
import { Icon } from "./icon";

interface FailureNoticeProps {
  failure: ApiFailure;
  /** Có thì hiện nút "Thử lại" khi lỗi có thể qua đi (mất mạng, máy chủ bận). */
  onRetry?: () => void;
  className?: string;
}

/** Thông báo lỗi gọi API bằng tiếng Việt theo `detail.code` (xem lib/api/errors.ts). */
export function FailureNotice({ failure, onRetry, className }: FailureNoticeProps) {
  const retry =
    onRetry && isRetryable(failure) ? (
      <Button variant="outline" size="sm" onClick={onRetry}>
        <Icon name="refresh" size="sm" />
        Thử lại
      </Button>
    ) : undefined;
  return (
    <Callout tone="err" role="alert" action={retry} className={className}>
      {failureMessage(failure)}
    </Callout>
  );
}
