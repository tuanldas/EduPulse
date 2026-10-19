import * as React from "react";

import { cn } from "@/lib/utils";

import { Icon, type IconName } from "./icon";

export type CalloutTone = "info" | "warn" | "ok" | "err";

const TONES: Record<CalloutTone, { box: string; icon: IconName }> = {
  info: { box: "bg-info-bg text-info-fg", icon: "info" },
  warn: { box: "bg-warn-bg text-warn-fg", icon: "warning" },
  ok: { box: "bg-ok-bg text-ok-fg", icon: "check_circle" },
  err: { box: "bg-err-bg text-err-fg", icon: "error" },
};

interface CalloutProps {
  tone?: CalloutTone;
  icon?: IconName;
  /** Câu in đậm đầu khối, ghép liền với nội dung phía sau. */
  title?: React.ReactNode;
  children?: React.ReactNode;
  /** `status`/`alert` cho thông báo xuất hiện sau thao tác; mặc định `note` cho lưu ý tĩnh. */
  role?: "note" | "status" | "alert";
  /** Nút hoặc liên kết đặt bên phải (vd. "Thử lại"). */
  action?: React.ReactNode;
  className?: string;
}

/**
 * Khối thông báo (st.info / warning / success / error của bản wireframe): nền nhạt + icon + chữ.
 * Màu chỉ là phụ; icon và câu chữ đã đủ nghĩa.
 */
export function Callout({ tone = "info", icon, title, children, role = "note", action, className }: CalloutProps) {
  const style = TONES[tone];
  return (
    <div
      role={role}
      className={cn(
        "flex flex-wrap items-start gap-x-3 gap-y-2 rounded-lg px-4 py-3.5 text-[15px] leading-[1.55] min-[641px]:flex-nowrap",
        style.box,
        className,
      )}
    >
      <Icon name={icon ?? style.icon} className="mt-px" />
      <div className="min-w-0 flex-1">
        {title ? <b className="font-semibold">{title} </b> : null}
        {children}
      </div>
      {/* Màn hình hẹp: nút xuống dòng riêng, thẳng hàng với chữ (icon 20px + khoảng cách 12px) thay vì ép chữ thành cột hẹp. */}
      {action ? <div className="shrink-0 max-[640px]:basis-full max-[640px]:pl-8">{action}</div> : null}
    </div>
  );
}
