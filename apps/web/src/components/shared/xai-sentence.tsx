import * as React from "react";

import type { HeadExplanation } from "@/lib/api/types";
import { sentenceTokens, type SentenceToken } from "@/lib/xai";
import { cn } from "@/lib/utils";

// Tô từ giải thích (docs/design-guidelines.md mục 2.2): từ ủng hộ = chữ đậm + số hạng ¹²³ + nền
// primary theo hạng; từ phản đối = chữ nghiêng + "−" + nền xám. Không chỉ dựa vào màu: có số hạng,
// kiểu chữ và chữ ẩn cho trình đọc màn hình. React tự escape văn bản nên không cần dangerouslySetInnerHTML.
const SUPPORT_BACKGROUND = ["bg-xai-3", "bg-xai-2", "bg-xai-1"] as const;

function supportBackground(rank: number): string {
  return SUPPORT_BACKGROUND[Math.min(rank, SUPPORT_BACKGROUND.length) - 1];
}

function Word({ token }: { token: SentenceToken }) {
  const { role } = token;
  if (role.kind === "none") return <>{token.text}</>;
  const support = role.kind === "support";
  return (
    <mark
      className={cn(
        "rounded-[5px] px-[3px] py-0.5 text-inherit",
        support ? cn("font-semibold", supportBackground(role.rank)) : "bg-xai-opp italic",
      )}
    >
      {token.text}
      <sup
        className={cn("ml-px text-[11px] font-bold", support ? "text-primary" : "text-muted-foreground")}
        aria-hidden="true"
      >
        {support ? role.rank : "−"}
      </sup>
      <span className="sr-only">{support ? ` (ủng hộ, hạng ${role.rank})` : " (phản đối)"}</span>
    </mark>
  );
}

interface XaiSentenceProps {
  head: HeadExplanation;
  /** Câu đã ẩn thông tin cá nhân (`display_text`): có thì khôi phục chữ hoa/thường của câu gốc. */
  originalText?: string;
  className?: string;
}

/** Câu đã tách từ với các từ ảnh hưởng nhất được tô nổi. */
export function XaiSentence({ head, originalText, className }: XaiSentenceProps) {
  const tokens = React.useMemo(
    () => sentenceTokens(head.tokens, head.support, head.oppose, originalText),
    [head.tokens, head.support, head.oppose, originalText],
  );
  return (
    <p className={cn("text-[21px] leading-[2.15] tracking-[0.005em] max-[640px]:text-[18px]", className)}>
      {tokens.map((token) => (
        <React.Fragment key={token.index}>
          {token.spaceBefore ? " " : null}
          <Word token={token} />
        </React.Fragment>
      ))}
    </p>
  );
}

/** Chú giải cách đọc câu tô (dưới câu giải thích). `label` là tên nhãn tiếng Việt đang giải thích. */
export function XaiLegend({ label, className }: { label: string; className?: string }) {
  return (
    <div className={cn("flex flex-wrap items-center gap-x-[18px] gap-y-1.5 text-[13px] text-muted-foreground", className)}>
      <span>
        <span className="inline-block rounded bg-xai-3 px-1.5 font-semibold text-foreground">từ</span> ủng hộ “
        {label}” (số 1 = mạnh nhất, màu đậm = mạnh)
      </span>
      <span>
        <span className="inline-block rounded bg-xai-opp px-1.5 text-foreground italic">từ</span> kéo về nhãn khác
        (chữ nghiêng, dấu −)
      </span>
      <span>Đơn vị là từ ghép, ví dụ “thầy cô”, “đúng giờ”.</span>
    </div>
  );
}
