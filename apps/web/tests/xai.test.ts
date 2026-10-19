import { describe, expect, it } from "vitest";

import type { HeadExplanation, TokenEffect } from "@/lib/api/types";
import { explainRows, restoreCasing, sentenceTokens } from "@/lib/xai";

function token(text: string, pWithout = 0.5): TokenEffect {
  return { text, delta: 0, p_without: pWithout };
}

// "Thầy hay vào lớp trễ và nghỉ không báo." — câu tự viết; "hay" là từ kéo về nhãn khác (phản đối).
const WORDS = ["Thầy", "hay", "vào lớp", "trễ", "và", "nghỉ", "không báo", "."].map((text) => token(text));

describe("ghép từ thành câu tô nổi", () => {
  it("gắn hạng cho từ ủng hộ theo thứ tự API và đánh dấu từ phản đối", () => {
    const result = sentenceTokens(WORDS, [6, 3, 5], [1]);
    expect(result[6].role).toEqual({ kind: "support", rank: 1 });
    expect(result[3].role).toEqual({ kind: "support", rank: 2 });
    expect(result[5].role).toEqual({ kind: "support", rank: 3 });
    expect(result[1].role).toEqual({ kind: "oppose" });
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
    tokens: [token("Thầy", 0.95), token("không báo", 0.46), token("trễ", 0.75), token("hay", 0.95)],
    support: [1, 2],
    oppose: [3],
  };

  it("liệt kê từ ủng hộ (mạnh trước) rồi từ phản đối, kèm thay đổi theo điểm", () => {
    const rows = explainRows(head);
    expect(rows.map((r) => [r.text, r.role, r.rank])).toEqual([
      ["không báo", "support", 1],
      ["trễ", "support", 2],
      ["hay", "oppose", null],
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
  const lower = ["thầy", "hay", "vào lớp", "trễ", "và", "nghỉ", "không báo", "."].map((text) => token(text));

  it("trả lại chữ hoa của câu gốc theo từng từ", () => {
    expect(restoreCasing(lower, "Thầy hay vào lớp trễ và Nghỉ không báo.")).toEqual([
      "Thầy",
      "hay",
      "vào lớp",
      "trễ",
      "và",
      "Nghỉ",
      "không báo",
      ".",
    ]);
  });

  it("từ không dò được thì giữ dạng của mô hình và không làm lệch các từ sau", () => {
    const result = restoreCasing(lower, "Thầy hay vào lớp muộn và Nghỉ không báo.");
    expect(result[3]).toBe("trễ");
    expect(result[5]).toBe("Nghỉ");
    expect(result[6]).toBe("không báo");
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
    const result = sentenceTokens(lower, [6], [], "Thầy hay vào lớp trễ và nghỉ không báo.");
    expect(result[0].text).toBe("Thầy");
    expect(result[6].role).toEqual({ kind: "support", rank: 1 });
  });
});
