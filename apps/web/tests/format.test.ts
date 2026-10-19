import { describe, expect, it } from "vitest";

import {
  formatBytes,
  formatDecimal,
  formatEta,
  formatNumber,
  formatPercent,
  formatPoints,
  formatScore,
} from "@/lib/format";

describe("định dạng số kiểu Việt", () => {
  it("dùng dấu chấm ngăn nghìn và dấu phẩy thập phân", () => {
    expect(formatNumber(1000)).toBe("1.000");
    expect(formatNumber(1409)).toBe("1.409");
    expect(formatNumber(0.018, 3)).toBe("0,018");
    expect(formatNumber(54.1, 1)).toBe("54,1");
  });

  it("đổi tỉ lệ sang phần trăm", () => {
    expect(formatPercent(0.9312)).toBe("93%");
    expect(formatPercent(0.541, 1)).toBe("54,1%");
    expect(formatPercent(1)).toBe("100%");
    expect(formatPercent(0)).toBe("0%");
  });

  it("không trả NaN hay Infinity ra giao diện", () => {
    expect(formatPercent(Number.NaN)).toBe("0%");
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe("0");
  });

  it("ghi điểm số 0..1 trên thang 100 và số thập phân giữ thang", () => {
    expect(formatScore(0.8561)).toBe("85,6");
    expect(formatDecimal(0.9607)).toBe("0,96");
    expect(formatDecimal(0.018, 3)).toBe("0,018");
  });

  it("ghi chênh lệch theo điểm có dấu trừ đúng, không dấu khi làm tròn về 0", () => {
    expect(formatPoints(-47.2)).toBe("−47");
    expect(formatPoints(2.4)).toBe("+2");
    expect(formatPoints(0.2)).toBe("0");
    expect(formatPoints(-0.2)).toBe("0");
    expect(formatPoints(12.34, 1)).toBe("+12,3");
  });

  it("ước lượng thời gian còn lại", () => {
    expect(formatEta(3)).toBe("vài giây");
    expect(formatEta(25)).toBe("khoảng 25 giây");
    expect(formatEta(27)).toBe("khoảng 25 giây");
    expect(formatEta(130)).toBe("khoảng 2 phút");
  });

  it("ghi dung lượng gọn", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1,5 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5 MB");
    expect(formatBytes(10 * 1024 * 1024)).toBe("10 MB");
  });
});
