import { describe, expect, it } from "vitest";

import { batchFailureMessage } from "@/lib/batch/batch-errors";
import { checkUploadFile } from "@/lib/batch/file-checks";

const MB = 1024 * 1024;

describe("checkUploadFile", () => {
  it("nhận CSV và XLSX trong giới hạn, không phân biệt hoa thường", () => {
    expect(checkUploadFile({ name: "gop-y.csv", size: 2 * MB }, 10)).toBeNull();
    expect(checkUploadFile({ name: "GOP-Y.XLSX", size: 10 * MB }, 10)).toBeNull();
  });

  it("từ chối đuôi file khác", () => {
    const failure = checkUploadFile({ name: "gop-y.pdf", size: 100 }, 10);
    expect(failure?.code).toBe("unsupported_file_type");
    expect(failure?.status).toBe(415);
    expect(checkUploadFile({ name: "gop-y.csv.exe", size: 100 }, 10)?.code).toBe(
      "unsupported_file_type",
    );
    expect(checkUploadFile({ name: "csv", size: 100 }, 10)?.code).toBe("unsupported_file_type");
  });

  it("từ chối file quá lớn và file rỗng", () => {
    expect(checkUploadFile({ name: "a.csv", size: 10 * MB + 1 }, 10)?.code).toBe("file_too_large");
    expect(checkUploadFile({ name: "a.csv", size: 0 }, 10)?.code).toBe("no_rows");
  });

  it("chưa biết giới hạn thì để máy chủ quyết định", () => {
    expect(checkUploadFile({ name: "a.csv", size: 500 * MB }, undefined)).toBeNull();
  });

  it("lỗi đi kèm câu tiếng Việt có ghi giới hạn", () => {
    const failure = checkUploadFile({ name: "a.csv", size: 11 * MB }, 10);
    expect(failure).not.toBeNull();
    expect(batchFailureMessage(failure!, { maxUploadMb: 10 })).toContain("10 MB");
  });
});
