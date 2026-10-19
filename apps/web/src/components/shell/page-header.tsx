import * as React from "react";

import { Icon, type IconName } from "@/components/shared/icon";

interface PageHeaderProps {
  icon: IconName;
  title: string;
  /** Một câu nói mục đích của trang. */
  description: React.ReactNode;
}

/** Đầu trang: tiêu đề h1 có icon trang + một câu mục đích (st.title + st.caption). */
export function PageHeader({ icon, title, description }: PageHeaderProps) {
  return (
    <header className="flex flex-col gap-2">
      <h1 className="flex items-center gap-2.5 text-[26px] leading-tight font-bold tracking-[-0.01em] min-[641px]:text-[2rem]">
        <Icon name={icon} className="size-[26px] text-primary min-[641px]:size-8" />
        {title}
      </h1>
      <p className="text-sm leading-[1.55] text-muted-foreground">{description}</p>
    </header>
  );
}
