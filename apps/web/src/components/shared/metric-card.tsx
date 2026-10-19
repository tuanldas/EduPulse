import * as React from "react";

import { cn } from "@/lib/utils";

import { Icon, type IconName } from "./icon";

interface MetricCardProps {
  label: React.ReactNode;
  value: React.ReactNode;
  /** Dòng giải thích ngắn dưới số (xám). */
  caption?: React.ReactNode;
  icon?: IconName;
  /** Chỗ cho phần phụ như chip chênh lệch ("điểm so với nền"). */
  children?: React.ReactNode;
  className?: string;
}

/** Ô chỉ số có viền (st.metric(border=True)): nhãn, số lớn 32px, chú thích. Dùng `dl` để đọc đúng cặp nhãn-giá trị. */
export function MetricCard({ label, value, caption, icon, children, className }: MetricCardProps) {
  return (
    <dl className={cn("flex min-w-0 flex-col gap-0.5 rounded-lg border border-border px-4 py-3.5", className)}>
      <dt className="flex items-center gap-1.5 text-sm">
        {icon ? <Icon name={icon} size="xs" className="text-muted-foreground" /> : null}
        {label}
      </dt>
      <dd className="text-[2rem] leading-[1.25] font-semibold tabular-nums">{value}</dd>
      {caption ? <dd className="text-sm text-muted-foreground">{caption}</dd> : null}
      {children}
    </dl>
  );
}
