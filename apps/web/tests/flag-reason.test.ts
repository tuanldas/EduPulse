import { describe, expect, it } from "vitest";

import { belowThreshold, reasonParts } from "@/components/analyze/flag-reason";
import type { AnalyzeResponse, FlagReason } from "@/lib/api/types";

import { fixture } from "./helpers/api-stub";

const lowConfidence: FlagReason = {
  code: "low_confidence",
  task: "topic",
  candidates: [
    { label: "lecturer", probability: 0.58 },
    { label: "facility", probability: 0.31 },
  ],
  message: "Chủ đề không chắc — phân vân giữa Giảng viên 58% và Cơ sở vật chất 31%",
};

describe("lý do gắn cờ bằng lời", () => {
  it("đầu ra không chắc: nêu tác vụ và hai lớp cao nhất kèm phần trăm", () => {
    expect(reasonParts(lowConfidence)).toEqual({
      title: "Chủ đề không chắc",
      detail: "— mô hình phân vân giữa Giảng viên (58%) và Cơ sở vật chất (31%).",
    });
  });

  it("chỉ có một ứng viên thì nói nghiêng về lớp đó", () => {
    const one = { ...lowConfidence, candidates: lowConfidence.candidates.slice(0, 1) };
    expect(reasonParts(one).detail).toBe("— mô hình nghiêng về Giảng viên (58%).");
    expect(reasonParts({ ...lowConfidence, candidates: [] }).detail).toBe("");
  });

  it("câu không dấu là lý do riêng (lý do thật do API trả về)", () => {
    const { response } = fixture<{ response: AnalyzeResponse }>("analyze-g13.json");
    const [reason] = response.flag.reasons;
    expect(reason.code).toBe("no_diacritics");
    expect(reasonParts(reason)).toEqual({
      title: "Câu không dấu",
      detail: "(luôn gắn cờ, kể cả khi độ tin cậy cao)",
    });
  });

  it("mã lạ thì dùng nguyên câu của máy chủ", () => {
    const unknown = { ...lowConfidence, code: "future_reason" } as unknown as FlagReason;
    expect(reasonParts(unknown)).toEqual({ title: lowConfidence.message, detail: "" });
  });

  it("đầu ra nào dưới ngưỡng cờ", () => {
    expect(belowThreshold([lowConfidence], "topic")).toBe(true);
    expect(belowThreshold([lowConfidence], "sentiment")).toBe(false);
    expect(belowThreshold([], "topic")).toBe(false);
  });
});
