import { describe, expect, it } from "vitest";

import { ICON_PATHS } from "@/components/shared/icon-paths";
import {
  LABEL_ORDER,
  PII_ORDER,
  PII_TAG,
  SENTIMENT_META,
  TASK_META,
  TOPIC_META,
  labelMeta,
} from "@/lib/labels";

describe("ánh xạ nhãn", () => {
  it("dịch nhãn cố định sang thuật ngữ tiếng Việt", () => {
    expect(labelMeta("sentiment", "negative")).toEqual({
      vi: "Tiêu cực",
      icon: "sentiment_dissatisfied",
      tone: "neg",
    });
    expect(labelMeta("sentiment", "neutral").vi).toBe("Trung tính");
    expect(labelMeta("sentiment", "positive").vi).toBe("Tích cực");
    expect(labelMeta("topic", "lecturer").vi).toBe("Giảng viên");
    expect(labelMeta("topic", "training_program").vi).toBe("Chương trình");
    expect(labelMeta("topic", "facility")).toMatchObject({ vi: "Cơ sở vật chất", tone: "csvc" });
    expect(labelMeta("topic", "others")).toMatchObject({ vi: "Khác", tone: "khac" });
  });

  it("nhãn lạ vẫn hiển thị bằng chính tên của nó", () => {
    expect(labelMeta("topic", "canteen")).toMatchObject({ vi: "canteen", icon: "help" });
  });

  it("thứ tự nhãn khớp bảng ánh xạ", () => {
    expect([...LABEL_ORDER.sentiment].sort()).toEqual(Object.keys(SENTIMENT_META).sort());
    expect([...LABEL_ORDER.topic].sort()).toEqual(Object.keys(TOPIC_META).sort());
  });

  it("mọi icon được nhắc tới đều có trong bộ icon", () => {
    const names = [
      ...Object.values(SENTIMENT_META),
      ...Object.values(TOPIC_META),
      ...Object.values(TASK_META),
    ].map((meta) => meta.icon);
    for (const name of names) expect(ICON_PATHS).toHaveProperty(name);
  });

  it("có thẻ thay thế cho từng loại thông tin cá nhân", () => {
    expect(Object.keys(PII_TAG).sort()).toEqual([...PII_ORDER].sort());
    expect(PII_TAG.name).toBe("[TÊN]");
    expect(PII_TAG.phone).toBe("[SĐT]");
  });
});
