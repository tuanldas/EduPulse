import * as React from "react";

import { AppSidebar } from "./app-sidebar";
import { MobileNav } from "./mobile-nav";
import { ThemeToggle } from "./theme-toggle";

/**
 * Khung ứng dụng: thanh bên 288px (> 640px) hoặc thanh đầu trang + Sheet (≤ 640px), vùng nội dung
 * có đệm 80px hai bên như wireframe, và liên kết "Bỏ qua điều hướng" cho bàn phím.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen min-[641px]:flex">
      <a
        href="#noi-dung"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-lg focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground"
      >
        Bỏ qua điều hướng
      </a>
      <AppSidebar className="max-[640px]:hidden" />
      <div className="min-w-0 flex-1">
        <MobileNav className="min-[641px]:hidden" />
        <header className="flex h-14 items-center justify-end px-5 max-[640px]:hidden">
          <ThemeToggle />
        </header>
        <main
          id="noi-dung"
          tabIndex={-1}
          className="mx-auto flex w-full max-w-[1200px] flex-col gap-4 px-4 pt-5 pb-24 outline-none min-[641px]:px-8 min-[641px]:pt-6 min-[641px]:pb-[120px] min-[1100px]:px-20"
        >
          {children}
        </main>
      </div>
    </div>
  );
}
