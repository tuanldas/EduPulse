"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

import { Icon } from "@/components/shared/icon";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

import { SidebarContent } from "./app-sidebar";
import { ThemeToggle } from "./theme-toggle";

/** Thanh đầu trang ở màn hình hẹp (≤ 640px): nút mở điều hướng (Sheet) + logo + công tắc giao diện. */
export function MobileNav({ className }: { className?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className={cn(
        "sticky top-0 z-20 flex h-14 items-center gap-2.5 border-b border-divider bg-background px-3",
        className,
      )}
    >
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetTrigger render={<Button variant="ghost" size="icon" aria-label="Mở điều hướng" />}>
          <Icon name="menu" />
        </SheetTrigger>
        <SheetContent side="left" className="w-80 max-w-[88vw] bg-muted px-4 pt-5 pb-6" showCloseButton>
          <SheetTitle className="sr-only">Điều hướng</SheetTitle>
          <SheetDescription className="sr-only">Chọn trang cần mở.</SheetDescription>
          <SidebarContent onNavigate={() => setOpen(false)} />
        </SheetContent>
      </Sheet>
      <Link href="/" aria-label="EduPulse — về trang Phân tích câu" className="rounded-md">
        <Image src="/logo-mark.svg" alt="EduPulse" width={28} height={28} className="size-7" />
      </Link>
      <div className="flex-1" />
      <ThemeToggle />
    </div>
  );
}
