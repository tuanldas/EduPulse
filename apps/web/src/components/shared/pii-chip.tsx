import * as React from "react";

import { PII_TAG } from "@/lib/labels";
import { cn } from "@/lib/utils";

/** Thẻ thông tin cá nhân đã ẩn, vd. [TÊN] [SĐT]: chữ đơn cách xám để dễ phân biệt với câu. */
export function PiiChip({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "rounded bg-neu-bg px-[5px] py-px font-mono text-[0.82em] font-medium whitespace-nowrap text-neu-fg",
        className,
      )}
    >
      {children}
    </span>
  );
}

// Khớp đúng các thẻ mà API dùng để thay thông tin cá nhân.
const TAG_PATTERN = new RegExp(
  `(${Object.values(PII_TAG)
    .map((tag) => tag.replace(/[[\]]/g, "\\$&"))
    .join("|")})`,
);

/** Câu đã ẩn thông tin cá nhân: các thẻ [TÊN] [SĐT] ... được bọc thành chip, còn lại giữ nguyên. */
export function PiiText({ text }: { text: string }) {
  const parts = text.split(TAG_PATTERN);
  return (
    <>
      {parts.map((part, index) =>
        index % 2 === 1 ? <PiiChip key={index}>{part}</PiiChip> : <React.Fragment key={index}>{part}</React.Fragment>,
      )}
    </>
  );
}

/** Đếm số lần xuất hiện của từng thẻ trong câu đã ẩn (`display_text`), để ghi "[TÊN] ×1 · [SĐT] ×1". */
export function countPiiTags(text: string): { tag: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const part of text.split(TAG_PATTERN)) {
    if (Object.values(PII_TAG).includes(part as (typeof PII_TAG)[keyof typeof PII_TAG])) {
      counts.set(part, (counts.get(part) ?? 0) + 1);
    }
  }
  return Object.values(PII_TAG)
    .filter((tag) => counts.has(tag))
    .map((tag) => ({ tag, count: counts.get(tag) ?? 0 }));
}
