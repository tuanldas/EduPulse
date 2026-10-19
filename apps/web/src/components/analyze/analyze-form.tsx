"use client";

import * as React from "react";

import { DemoUnavailableNotice } from "@/components/shared/demo-unavailable-notice";
import { Icon } from "@/components/shared/icon";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { AppConfig } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";

import { SampleChips } from "./sample-chips";

interface AnalyzeFormProps {
  text: string;
  onTextChange: (text: string) => void;
  onSubmit: () => void;
  onClear: () => void;
  loading: boolean;
  /** Giới hạn ký tự của một câu; chưa biết (chưa nạp cấu hình) thì không chặn ở trình duyệt. */
  maxChars: number | undefined;
  samples: AppConfig["ui_samples"];
  /** `demo_available`: false thì ẩn chip và báo dữ liệu demo chưa sẵn. */
  demoAvailable: boolean | undefined;
  demoReason: string | null | undefined;
  activeText: string | null;
  onPickSample: (text: string) => void;
  compare: boolean;
  onCompareChange: (compare: boolean) => void;
}

/** Ô nhập câu, câu mẫu, nút Phân tích / Xoá và công tắc so sánh mô hình (mặc định tắt). */
export function AnalyzeForm({
  text,
  onTextChange,
  onSubmit,
  onClear,
  loading,
  maxChars,
  samples,
  demoAvailable,
  demoReason,
  activeText,
  onPickSample,
  compare,
  onCompareChange,
}: AnalyzeFormProps) {
  const canSubmit = text.trim().length > 0 && !loading;

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && canSubmit) {
      event.preventDefault();
      onSubmit();
    }
  }

  return (
    <form
      className="flex flex-col gap-4"
      onSubmit={(event) => {
        event.preventDefault();
        if (canSubmit) onSubmit();
      }}
    >
      <div>
        <div className="mb-1.5 flex items-center gap-1.5 text-sm">
          <label htmlFor="cau-gop-y">Câu góp ý</label>
          <Tooltip>
            <TooltipTrigger
              render={
                <button
                  type="button"
                  aria-label="Gợi ý khi nhập câu"
                  className="rounded-full text-muted-foreground outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
                />
              }
            >
              <Icon name="help" size="xs" />
            </TooltipTrigger>
            <TooltipContent>
              {maxChars ? `Tối đa ${formatNumber(maxChars)} ký tự. ` : ""}Nhấn Ctrl/⌘ + Enter để phân tích. Câu bạn nhập
              không được lưu.
            </TooltipContent>
          </Tooltip>
        </div>
        <div className="relative">
          <Textarea
            id="cau-gop-y"
            value={text}
            maxLength={maxChars}
            rows={3}
            placeholder="Nhập một câu góp ý của sinh viên…"
            onChange={(event) => onTextChange(event.target.value)}
            onKeyDown={handleKeyDown}
            className="pb-7"
          />
          <span
            aria-hidden="true"
            className="pointer-events-none absolute right-3 bottom-1.5 text-xs text-muted-foreground tabular-nums"
          >
            {formatNumber(text.length)}
            {maxChars ? `/${formatNumber(maxChars)}` : ""}
          </span>
        </div>
      </div>

      {demoAvailable === false ? (
        <DemoUnavailableNotice reason={demoReason} />
      ) : samples.length > 0 ? (
        <SampleChips samples={samples} activeText={activeText} onPick={onPickSample} />
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" disabled={!canSubmit}>
          <Icon name="play_arrow" size="sm" />
          {loading ? "Đang phân tích…" : "Phân tích"}
        </Button>
        <Button type="button" variant="ghost" onClick={onClear}>
          <Icon name="close" size="sm" />
          Xoá
        </Button>
        <div className="flex-1" />
        <label className="inline-flex cursor-pointer items-center gap-2.5 text-[15px]">
          <Switch checked={compare} onCheckedChange={onCompareChange} />
          So sánh với mô hình cơ sở
        </label>
      </div>
    </form>
  );
}
