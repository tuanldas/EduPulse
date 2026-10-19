import { describe, expect, it } from "vitest";

import { anchorId } from "@/lib/anchor";

describe("anchorId", () => {
  it("bỏ dấu # đầu và giữ nguyên id thường", () => {
    expect(anchorId("#giay-phep")).toBe("giay-phep");
    expect(anchorId("giay-phep")).toBe("giay-phep");
  });

  it("giải mã chuỗi phần trăm UTF-8 của tiếng Việt", () => {
    expect(anchorId("#gi%E1%BA%A5y-ph%C3%A9p")).toBe("giấy-phép");
  });

  it("neo hỏng thì dùng nguyên văn thay vì ném URIError", () => {
    expect(() => anchorId("#%E0%A4%A")).not.toThrow();
    expect(anchorId("#%E0%A4%A")).toBe("%E0%A4%A");
    expect(anchorId("#%")).toBe("%");
    expect(anchorId("#%zz")).toBe("%zz");
  });

  it("không có neo thì trả về chuỗi rỗng", () => {
    expect(anchorId("")).toBe("");
    expect(anchorId("#")).toBe("");
  });
});
