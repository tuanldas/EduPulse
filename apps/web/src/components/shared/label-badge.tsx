import { Badge } from "@/components/ui/badge";
import { formatPercent } from "@/lib/format";
import { labelMeta, type Task } from "@/lib/labels";
import { cn } from "@/lib/utils";

import { Icon } from "./icon";

interface LabelBadgeProps {
  task: Task;
  /** Nhãn của API (`negative`, `lecturer`, ...). */
  label: string;
  /** Độ tin cậy 0..1; có thì ghi kèm ` · 93%`. */
  confidence?: number;
  size?: "default" | "lg";
  className?: string;
}

/** Nhãn cảm xúc hoặc chủ đề: icon + tên tiếng Việt (+ độ tin cậy), màu theo khoá ngữ nghĩa. */
export function LabelBadge({ task, label, confidence, size = "default", className }: LabelBadgeProps) {
  const meta = labelMeta(task, label);
  return (
    <Badge variant={meta.tone} size={size} className={cn(className)}>
      <Icon name={meta.icon} size="xs" />
      {meta.vi}
      {confidence !== undefined ? ` · ${formatPercent(confidence)}` : null}
    </Badge>
  );
}
