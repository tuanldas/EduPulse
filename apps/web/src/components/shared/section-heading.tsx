import * as React from "react";

import { cn } from "@/lib/utils";

import { Icon, type IconName } from "./icon";

interface HeadingProps {
  icon?: IconName;
  id?: string;
  children: React.ReactNode;
  className?: string;
}

/** Tiêu đề mục (h2, 24/32 · 600) với icon xám phía trước. `id` làm đích neo cho mục lục. */
export function SectionHeading({ icon, id, children, className }: HeadingProps) {
  return (
    <h2 id={id} className={cn("flex items-center gap-2 text-2xl leading-8 font-semibold", className)}>
      {icon ? <Icon name={icon} className="text-muted-foreground" /> : null}
      {children}
    </h2>
  );
}

/** Tiêu đề khối (h3, 20/28 · 600). */
export function SubHeading({ icon, id, children, className }: HeadingProps) {
  return (
    <h3 id={id} className={cn("flex items-center gap-2 text-xl leading-7 font-semibold", className)}>
      {icon ? <Icon name={icon} className="text-muted-foreground" /> : null}
      {children}
    </h3>
  );
}

/** Dòng chú thích 14px màu phụ (st.caption). */
export function Caption({
  children,
  className,
  ...props
}: React.ComponentProps<"p">) {
  return (
    <p className={cn("text-sm leading-[1.55] text-muted-foreground", className)} {...props}>
      {children}
    </p>
  );
}
