"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Icon } from "@/components/shared/icon";
import { buttonVariants } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { retryAppConfig, useAppConfig } from "@/lib/app-config";
import { cn } from "@/lib/utils";

import { isActivePath, NAV_ITEMS } from "./nav-items";

const LINK_BUTTON = cn(buttonVariants({ variant: "outline", size: "sm" }), "w-full text-foreground");

function ModelInfo() {
  const state = useAppConfig();
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-xs font-semibold tracking-[0.02em] text-muted-foreground uppercase">Mô hình đang dùng</div>
      <div className="text-sm leading-[1.45]">PhoBERT-base-v2 đa nhiệm · ONNX int8</div>
      {state.status === "ready" ? (
        <>
          <div className="text-[13px] leading-normal text-muted-foreground">
            <code>
              {state.config.model.model_version} · {state.config.model.bundle_version}
            </code>
          </div>
          {state.config.model.stage === "smoke" ? (
            <div className="text-[13px] leading-normal text-muted-foreground">
              Bản smoke: số đo chưa phải kết quả cuối.
            </div>
          ) : null}
        </>
      ) : state.status === "error" ? (
        <div className="text-[13px] leading-normal text-muted-foreground">
          Chưa kết nối được máy chủ.{" "}
          <button type="button" className="text-link underline underline-offset-2" onClick={() => void retryAppConfig()}>
            Thử lại
          </button>
        </div>
      ) : (
        <Skeleton className="h-4 w-40" />
      )}
    </div>
  );
}

function SourceLink() {
  const state = useAppConfig();
  const sourceUrl = state.status === "ready" ? state.config.source_url : "";
  if (sourceUrl) {
    return (
      <a href={sourceUrl} target="_blank" rel="noopener noreferrer" className={LINK_BUTTON}>
        <Icon name="code" size="sm" />
        Mã nguồn · AGPL-3.0
      </a>
    );
  }
  // Chưa cấu hình liên kết mã nguồn thì dẫn tới mục giấy phép của trang Mô hình & đạo đức.
  return (
    <Link href="/model/#giay-phep" className={LINK_BUTTON}>
      <Icon name="gavel" size="sm" />
      Giấy phép · AGPL-3.0
    </Link>
  );
}

/** Nội dung thanh bên: logo, điều hướng 4 trang, mô hình đang dùng, lưu ý, mã nguồn. Dùng cả ở Sheet di động. */
export function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <div className="flex h-full flex-col gap-3.5">
      <div className="mb-1.5 flex min-h-10 items-center">
        <Link href="/" onClick={onNavigate} aria-label="EduPulse — về trang Phân tích câu" className="rounded-md">
          <Image src="/logo.svg" alt="EduPulse" width={127} height={32} priority className="h-8 w-auto" />
        </Link>
      </div>
      <nav aria-label="Điều hướng chính" className="flex flex-col gap-0.5">
        {NAV_ITEMS.map((item) => {
          const active = isActivePath(pathname, item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onNavigate}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex min-h-10 items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[15px] text-foreground outline-none hover:bg-accent/60 focus-visible:ring-3 focus-visible:ring-ring/50",
                active && "bg-accent font-semibold hover:bg-accent",
              )}
            >
              <Icon name={item.icon} className={active ? "text-foreground" : "text-muted-foreground"} />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <Separator className="my-1" />
      <ModelInfo />
      <div className="flex gap-2 rounded-lg border border-border bg-background px-3 py-2.5 text-[13px] leading-normal">
        <Icon name="info" size="sm" className="mt-px text-flag-fg" />
        <span>
          <b className="font-semibold">AI gợi ý — cần người xem xét.</b> Không dùng để đánh giá hay kỷ luật cá nhân.
        </span>
      </div>
      <SourceLink />
      <p className="mt-auto text-xs leading-normal text-muted-foreground">
        Không lưu dữ liệu bạn nhập.
        <br />
        Đồ án học thuật · dữ liệu UIT-VSFC.
      </p>
    </div>
  );
}

/**
 * Thanh bên cố định ở màn hình rộng (> 640px). Cột ngoài kéo dài hết chiều cao trang để nền thanh
 * bên không đứt khi trang dài; phần trong dính ở đầu màn hình và cuộn riêng khi màn hình thấp.
 */
export function AppSidebar({ className }: { className?: string }) {
  return (
    <div className={cn("w-72 shrink-0 border-r border-divider bg-muted", className)}>
      <aside className="sticky top-0 h-screen overflow-y-auto px-4 pt-5 pb-6">
        <SidebarContent />
      </aside>
    </div>
  );
}
