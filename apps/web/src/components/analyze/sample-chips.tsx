"use client";

import type { AppConfig } from "@/lib/api/types";
import { cn } from "@/lib/utils";

type Sample = AppConfig["ui_samples"][number];

interface SampleChipsProps {
  samples: readonly Sample[];
  /** Câu đang được phân tích/hiển thị, để tô chip tương ứng. */
  activeText: string | null;
  onPick: (text: string) => void;
}

/** Câu mẫu dạng chip: chọn một chip là điền câu vào ô nhập và phân tích luôn. */
export function SampleChips({ samples, activeText, onPick }: SampleChipsProps) {
  return (
    <div role="group" aria-labelledby="sample-chips-label" className="flex flex-col gap-1.5">
      <div id="sample-chips-label" className="text-sm">
        Hoặc thử câu mẫu (UIT-VSFC)
      </div>
      <div className="flex flex-wrap gap-2">
        {samples.map((sample) => {
          const active = sample.text === activeText;
          return (
            <button
              key={sample.text}
              type="button"
              title={sample.text}
              aria-pressed={active}
              onClick={() => onPick(sample.text)}
              className={cn(
                "inline-flex min-h-[34px] max-w-full items-center gap-1.5 rounded-full border border-border bg-background px-3 py-[5px] text-sm leading-[1.35] outline-none hover:bg-accent/60 focus-visible:ring-3 focus-visible:ring-ring/50",
                active && "border-primary bg-primary-soft font-medium text-primary hover:bg-primary-soft",
              )}
            >
              <span className="block max-w-[36ch] truncate">{sample.text}</span>
              {sample.no_diacritics ? (
                <span className="shrink-0 text-xs text-muted-foreground">· không dấu</span>
              ) : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}
