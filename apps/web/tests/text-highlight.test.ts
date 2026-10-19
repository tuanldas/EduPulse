import { describe, expect, it } from "vitest";

import { emphasizeSummary, splitByTerms } from "@/lib/dashboard/text-highlight";

function marked(segments: { text: string; mark: boolean }[]) {
  return segments.filter((segment) => segment.mark).map((segment) => segment.text);
}

function joined(segments: { text: string }[]) {
  return segments.map((segment) => segment.text).join("");
}

describe("splitByTerms", () => {
  it("tô đúng các từ khoá và giữ nguyên phần còn lại", () => {
    const text = "Ngồi cuối lớp khó đọc chữ vì máy chiếu quá mờ.";
    const segments = splitByTerms(text, ["máy chiếu", "mờ"]);
    expect(marked(segments)).toEqual(["máy chiếu", "mờ"]);
    expect(joined(segments)).toBe(text);
  });

  it("không phân biệt hoa/thường và giữ chữ gốc", () => {
    const segments = splitByTerms("Máy chiếu hôm nay hơi nhoè.", ["máy chiếu"]);
    expect(marked(segments)).toEqual(["Máy chiếu"]);
  });

  it("chỉ khớp nguyên từ, kể cả với chữ có dấu", () => {
    expect(marked(splitByTerms("Mapping chạy rất lâu.", ["map"]))).toEqual([]);
    expect(marked(splitByTerms("Bài tập mờ ảo.", ["mờ"]))).toEqual(["mờ"]);
    expect(marked(splitByTerms("Ánh sáng nhờ rèm.", ["nhờ"]))).toEqual(["nhờ"]);
    expect(marked(splitByTerms("Ánh sáng nhờ rèm.", ["hờ"]))).toEqual([]);
  });

  it("từ dài được ưu tiên hơn từ ngắn nằm trong nó", () => {
    const segments = splitByTerms("Phòng máy rất nóng.", ["phòng", "phòng máy"]);
    expect(marked(segments)).toEqual(["Phòng máy"]);
  });

  it("khớp mọi lần xuất hiện", () => {
    const segments = splitByTerms("Lớp em dùng máy chiếu đời cũ, cần đổi sang máy chiếu loại mới.", ["máy chiếu"]);
    expect(marked(segments)).toEqual(["máy chiếu", "máy chiếu"]);
  });

  it("xử lý ký tự đặc biệt của biểu thức chính quy như chữ thường", () => {
    const segments = splitByTerms("Giá (nâng cấp) +10% sao?", ["(nâng cấp)", "+10%"]);
    expect(marked(segments)).toEqual(["(nâng cấp)", "+10%"]);
    expect(joined(segments)).toBe("Giá (nâng cấp) +10% sao?");
    // Dấu `.` và `*` không được hiểu là ký tự thay thế.
    expect(marked(splitByTerms("abc a.c", ["a.c"]))).toEqual(["a.c"]);
    expect(marked(splitByTerms("abc ab", ["a*"]))).toEqual([]);
  });

  it("không có từ khoá thì trả nguyên văn", () => {
    expect(splitByTerms("Thầy dạy dễ hiểu.", [])).toEqual([
      { text: "Thầy dạy dễ hiểu.", mark: false },
    ]);
    expect(splitByTerms("Thầy dạy dễ hiểu.", ["", "  "])).toEqual([
      { text: "Thầy dạy dễ hiểu.", mark: false },
    ]);
  });

  it("chuẩn hoá Unicode trước khi so khớp", () => {
    const decomposed = "máy chiếu mờ".normalize("NFD");
    const segments = splitByTerms(decomposed, ["máy chiếu"]);
    expect(marked(segments)).toEqual(["máy chiếu"]);
  });
});

describe("emphasizeSummary", () => {
  it("in đậm phần trăm đầu tiên và cụm 'khoảng N điểm'", () => {
    const summary =
      "93% góp ý về Chương trình là tiêu cực (182/195 câu), cao hơn mức nền 78% khoảng 15 điểm. Cao ở cả 6 môn.";
    const segments = emphasizeSummary(summary);
    expect(marked(segments)).toEqual(["93%", "khoảng 15 điểm"]);
    expect(joined(segments)).toBe(summary);
  });

  it("nhận 'gấp đôi' và 'gấp N lần'", () => {
    expect(
      marked(emphasizeSummary("9% góp ý nói về Cơ sở vật chất (90/1.000 câu), gấp đôi mức thường gặp 4,4%.")),
    ).toEqual(["9%", "gấp đôi"]);
    expect(marked(emphasizeSummary("Nhắc gấp 2,5 lần mức thường gặp."))).toEqual(["gấp 2,5 lần"]);
  });

  it("phần trăm thập phân được nhận trọn", () => {
    expect(marked(emphasizeSummary("Tiêu cực 54,1% so với nền 46,6%."))).toEqual(["54,1%"]);
  });

  it("câu không có số cần nhấn mạnh giữ nguyên một đoạn", () => {
    expect(emphasizeSummary("Tóm tắt này không có số nào cần in đậm.")).toEqual([
      { text: "Tóm tắt này không có số nào cần in đậm.", mark: false },
    ]);
  });

  it("chuỗi rỗng không tạo đoạn nào", () => {
    expect(emphasizeSummary("")).toEqual([]);
  });
});
