import { Callout } from "@/components/shared/callout";
import { Caption } from "@/components/shared/section-heading";
import { PiiChip, PiiText, countPiiTags } from "@/components/shared/pii-chip";
import { PII_TAG } from "@/lib/labels";
import { formatNumber } from "@/lib/format";

/** Trạng thái đầu: chưa nhập câu nào. */
export function EmptyState() {
  return (
    <Callout tone="info" icon="edit_note" role="status">
      Nhập một câu hoặc chọn câu mẫu ở trên để bắt đầu. Mỗi câu mất khoảng 1–2 giây, kể cả phần giải thích.
    </Callout>
  );
}

/** Câu gõ không dấu: kết quả có thể kém chính xác, luôn được gắn cờ. */
export function NoDiacriticsNotice() {
  return (
    <Callout tone="warn" icon="spellcheck" title="Câu có vẻ được gõ không dấu." role="status">
      Mô hình học từ tiếng Việt có dấu nên kết quả có thể kém chính xác. Hãy gõ lại có dấu nếu được.
    </Callout>
  );
}

/** Câu không còn nội dung để phân tích sau khi ẩn thông tin cá nhân và bỏ ký hiệu (mã `no_content`). */
export function NoContentNotice() {
  return (
    <Callout tone="warn" icon="shield_person" title="Không còn nội dung để phân tích." role="status">
      Câu chỉ gồm thông tin cá nhân hoặc ký hiệu nên đã được ẩn hết. Hãy nhập câu có nội dung góp ý.
    </Callout>
  );
}

/** Câu dài hơn giới hạn từ của giải thích: vẫn phân loại bình thường, chỉ không tô từ. */
export function LongSentenceNotice({ nWords, maxWords }: { nWords: number; maxWords: number | undefined }) {
  return (
    <Callout tone="info" icon="short_text" role="status">
      <b className="font-semibold">
        Câu dài {formatNumber(nWords)} từ{maxWords ? ` (> ${formatNumber(maxWords)})` : ""}
      </b>{" "}
      nên không tô từ để giữ tốc độ. Kết quả phân loại vẫn dùng được; câu dài không tự bị gắn cờ.
    </Callout>
  );
}

/**
 * Câu có thông tin cá nhân đã bị ẩn: nói rõ ẩn những gì, cho xem câu đã ẩn, và nhắc giới hạn của
 * việc nhận diện tên (best effort).
 */
export function PiiNotice({ displayText }: { displayText: string }) {
  const found = countPiiTags(displayText);
  const total = found.reduce((sum, item) => sum + item.count, 0);
  return (
    <div className="flex flex-col gap-3">
      <Callout tone="info" icon="shield_person" role="status">
        <b className="font-semibold">Đã ẩn {formatNumber(total)} thông tin cá nhân</b> trước khi phân tích:{" "}
        {found.map((item, index) => (
          <span key={item.tag}>
            {index > 0 ? " · " : ""}
            <PiiChip>{item.tag}</PiiChip> ×{item.count}
          </span>
        ))}
        . Kết quả và file xuất chỉ chứa câu đã ẩn.
      </Callout>
      <p className="text-[17px] leading-[1.9]">
        <PiiText text={displayText} />
      </p>
      <Caption>
        Loại được ẩn: <PiiChip>{PII_TAG.phone}</PiiChip> <PiiChip>{PII_TAG.email}</PiiChip>{" "}
        <PiiChip>{PII_TAG.student_id}</PiiChip> <PiiChip>{PII_TAG.national_id}</PiiChip> <PiiChip>{PII_TAG.url}</PiiChip>{" "}
        theo quy tắc; <PiiChip>{PII_TAG.name}</PiiChip> chỉ ở mức cố gắng tối đa (best effort), tên viết thường có thể sót →
        nên tự ẩn danh trước.
      </Caption>
    </div>
  );
}
