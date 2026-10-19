import { Progress, ProgressLabel } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface ConfidenceBarProps {
  /** Độ tin cậy đã hiệu chỉnh, 0..1. */
  value: number;
  /** Đúng khi độ tin cậy của đầu ra này nằm dưới ngưỡng cờ. */
  flagged?: boolean;
  className?: string;
}

/** Thanh "Độ tin cậy (đã hiệu chỉnh)" kèm câu cho biết đã đạt ngưỡng cờ hay chưa. */
export function ConfidenceBar({ value, flagged = false, className }: ConfidenceBarProps) {
  const percent = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <Progress
      value={percent}
      className={cn(className)}
      indicatorClassName={flagged ? "bg-flag" : undefined}
    >
      <ProgressLabel className="text-sm text-muted-foreground">
        Độ tin cậy (đã hiệu chỉnh) ·{" "}
        {flagged ? <b className="font-semibold text-flag-fg">dưới ngưỡng cờ</b> : "đạt ngưỡng"}
      </ProgressLabel>
    </Progress>
  );
}
