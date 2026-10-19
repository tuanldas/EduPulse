import { describe, expect, it } from "vitest";

import type { HeadExplanation, TokenEffect } from "@/lib/api/types";
import { explainRows, restoreCasing, sentenceTokens } from "@/lib/xai";

function token(text: string, pWithout = 0.5): TokenEffect {
  return { text, delta: 0, p_without: pWithout };
}

const WORDS = ["[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "."].map((text) => token(text));

describe("ghép từ thành câu tô nổi", () => {
  it("gắn hạng cho từ ủng hộ theo thứ tự API và đánh dấu từ phản đối", () => {
    const result = sentenceTokens(WORDS, [5, 6, 1], [3]);
    expect(result[5].role).toEqual({ kind: "support", rank: 1 });
    expect(result[6].role).toEqual({ kind: "support", rank: 2 });
    expect(result[1].role).toEqual({ kind: "support", rank: 3 });
    expect(result[3].role).toEqual({ kind: "oppose" });
    expect(result[0].role).toEqual({ kind: "none" });
  });

  it("chèn dấu cách giữa các từ nhưng dấu câu dính vào từ trước", () => {
    const result = sentenceTokens(WORDS, [], []);
    expect(result.map((t) => t.spaceBefore)).toEqual([false, true, true, true, true, true, true, false]);
  });

  it("emoticon và thẻ [TÊN] vẫn có dấu cách, dấu hai chấm thì dính", () => {
    const tokens = ["Cô", "[TÊN]", "dạy", "hay", ":", "tốt", ":)"].map((text) => token(text));
    const result = sentenceTokens(tokens, [], []);
    expect(result.map((t) => t.spaceBefore)).toEqual([false, true, true, true, false, true, true]);
  });

  it("không chèn dấu cách sau dấu mở ngoặc", () => {
    const tokens = ["Giờ", "(", "sáng", ")", "."].map((text) => token(text));
    const result = sentenceTokens(tokens, [], []);
    expect(result.map((t) => t.spaceBefore)).toEqual([false, true, false, false, false]);
  });

  it("câu rỗng cho ra danh sách rỗng", () => {
    expect(sentenceTokens([], [], [])).toEqual([]);
  });
});

describe("dòng bảng giải thích", () => {
  const head: HeadExplanation = {
    label: "negative",
    probability: 0.93,
    tokens: [token("Có", 0.95), token("không", 0.46), token("đúng giờ", 0.75), token("đến", 0.95)],
    support: [1, 2],
    oppose: [3],
  };

  it("liệt kê từ ủng hộ (mạnh trước) rồi từ phản đối, kèm thay đổi theo điểm", () => {
    const rows = explainRows(head);
    expect(rows.map((r) => [r.text, r.role, r.rank])).toEqual([
      ["không", "support", 1],
      ["đúng giờ", "support", 2],
      ["đến", "oppose", null],
    ]);
    expect(rows[0].pFull).toBe(0.93);
    expect(rows[0].pWithout).toBe(0.46);
    expect(rows[0].changePoints).toBeCloseTo(-47, 5);
    expect(rows[2].changePoints).toBeCloseTo(2, 5);
  });

  it("bỏ qua chỉ số nằm ngoài danh sách từ", () => {
    expect(explainRows({ ...head, support: [9], oppose: [] })).toEqual([]);
  });
});

describe("khôi phục chữ hoa/thường của câu gốc", () => {
  const lower = ["[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "[đã gỡ câu UIT-VSFC]", "."].map((text) => token(text));

  it("trả lại chữ hoa của câu gốc theo từng từ", () => {
    expect(restoreCasing(lower, "[đã gỡ câu UIT-VSFC].")).toEqual([
      "Có",
      "nhiều",
      "Thầy cô",
      "đến",
      "lớp",
      "không",
      "đúng giờ",
      ".",
    ]);
  });

  it("từ không dò được thì giữ dạng của mô hình và không làm lệch các từ sau", () => {
    const result = restoreCasing(lower, "Có nhiều thầy cô tới lớp Không đúng giờ.");
    expect(result[3]).toBe("đến");
    expect(result[5]).toBe("Không");
    expect(result[6]).toBe("đúng giờ");
  });

  it("chỉ khớp từ trọn vẹn, không khớp giữa một từ khác", () => {
    const tokens = ["bạn", "an"].map((text) => token(text));
    expect(restoreCasing(tokens, "Bạn An")).toEqual(["Bạn", "An"]);
    expect(restoreCasing([token("an")], "Bạn đang An")).toEqual(["An"]);
  });

  it("giữ nguyên thẻ [TÊN] và câu rỗng", () => {
    expect(restoreCasing([token("cô"), token("[TÊN]")], "Cô [TÊN] dạy hay")).toEqual(["Cô", "[TÊN]"]);
    expect(restoreCasing([token("có")], "")).toEqual(["có"]);
  });

  it("sentenceTokens dùng câu gốc khi có", () => {
    const result = sentenceTokens(lower, [5], [], "[đã gỡ câu UIT-VSFC].");
    expect(result[0].text).toBe("Có");
    expect(result[5].role).toEqual({ kind: "support", rank: 1 });
  });
});
