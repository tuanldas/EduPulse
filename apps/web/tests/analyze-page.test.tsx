import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { AnalyzePage } from "@/components/analyze/analyze-page";
import type { AnalyzeResponse, ExplainResponse } from "@/lib/api/types";
import { resetAppConfig } from "@/lib/app-config";
import { labelMeta } from "@/lib/labels";

import { callsTo, deferred, fixture, json, stubApi } from "./helpers/api-stub";

// Phản hồi trong tests/fixtures/ do `pnpm record:fixtures` ghi từ API thật trên các câu tự viết
// (golden_sentences.json): g02 phân tích thường, g08 có từ được tô, g13 không dấu, g14 có số điện
// thoại, g17 câu dài hơn giới hạn từ giải thích.
interface Recorded<T> {
  text: string;
  response: T;
}
const config = fixture("config.json");
const analysis = (id: string) => fixture<Recorded<AnalyzeResponse>>(`analyze-${id}.json`);
const explanation = (id: string) => fixture<Recorded<ExplainResponse>>(`explain-${id}.json`);

/** Dán câu vào ô nhập rồi bấm "Phân tích". */
async function submit(text: string) {
  const user = userEvent.setup();
  await user.click(await screen.findByLabelText("Câu góp ý"));
  await user.paste(text);
  await user.click(screen.getByRole("button", { name: "Phân tích" }));
  return user;
}

beforeEach(() => {
  resetAppConfig();
});

describe("trang Phân tích câu", () => {
  it("hiện nhãn ngay khi /api/analyze trả về, giải thích tải sau mà không chặn kết quả", async () => {
    const slowExplain = deferred<Response>();
    const { text, response } = analysis("g08");
    stubApi({
      "GET /api/config": () => json(config),
      "POST /api/analyze": () => json(response),
      "POST /api/explain": () => slowExplain.promise,
    });
    const { container } = render(<AnalyzePage />);
    await submit(text);

    await screen.findByRole("heading", { name: "Kết quả" });
    expect(screen.getAllByText(labelMeta("sentiment", response.sentiment.label).vi).length).toBeGreaterThan(0);
    expect(screen.getAllByText(labelMeta("topic", response.topic.label).vi).length).toBeGreaterThan(0);
    expect(screen.getByLabelText("Đang tính giải thích")).toBeInTheDocument();
    expect(container.querySelectorAll("mark")).toHaveLength(0);

    slowExplain.resolve(json(explanation("g08").response));
    await waitFor(() => expect(screen.queryByLabelText("Đang tính giải thích")).toBeNull());
    // g08: 3 từ ủng hộ và 1 từ phản đối ở đầu ra cảm xúc.
    expect(container.querySelectorAll("mark")).toHaveLength(4);
    expect(callsTo("POST /api/explain")).toHaveLength(1);
  });

  it("nói rõ khi không có từ nào đủ lớn để tô", async () => {
    const { text, response } = analysis("g02");
    stubApi({
      "GET /api/config": () => json(config),
      "POST /api/analyze": () => json(response),
      "POST /api/explain": () => json(explanation("g02").response),
    });
    const { container } = render(<AnalyzePage />);
    await submit(text);
    await screen.findByText(/Không có từ nào ảnh hưởng đủ lớn/);
    expect(container.querySelectorAll("mark")).toHaveLength(0);
  });

  it("câu dài hơn giới hạn từ: không gọi /api/explain, báo không tô từ và vẫn hiện kết quả", async () => {
    const { text, response } = analysis("g17");
    expect(response.xai_available).toBe(false);
    stubApi({
      "GET /api/config": () => json(config),
      "POST /api/analyze": () => json(response),
      "POST /api/explain": () => json({ detail: { code: "xai_too_long", message: "x" } }, 422),
    });
    render(<AnalyzePage />);
    await submit(text);
    await screen.findByText(/nên không tô từ để giữ tốc độ/);
    expect(screen.getByRole("heading", { name: "Kết quả" })).toBeInTheDocument();
    expect(callsTo("POST /api/explain")).toHaveLength(0);
  });

  it("câu gõ không dấu: cảnh báo và gắn cờ với lý do riêng", async () => {
    const { text, response } = analysis("g13");
    stubApi({
      "GET /api/config": () => json(config),
      "POST /api/analyze": () => json(response),
      "POST /api/explain": () => json(explanation("g02").response),
    });
    render(<AnalyzePage />);
    await submit(text);
    await screen.findByText(/Câu có vẻ được gõ không dấu/);
    expect(screen.getByText("Cần xem xét")).toBeInTheDocument();
    expect(screen.getByText("Câu không dấu")).toBeInTheDocument();
  });

  it("câu có thông tin cá nhân: báo đã ẩn và chỉ hiện câu đã ẩn", async () => {
    const { text, response } = analysis("g14");
    expect(response.pii_types).toContain("phone");
    stubApi({
      "GET /api/config": () => json(config),
      "POST /api/analyze": () => json(response),
      "POST /api/explain": () => json(explanation("g02").response),
    });
    render(<AnalyzePage />);
    await submit(text);
    await screen.findByText(/Đã ẩn 1 thông tin cá nhân/);
    expect(screen.getAllByText("[SĐT]").length).toBeGreaterThan(0);
    // Số thật chỉ còn trong ô nhập của chính người dùng, không xuất hiện ở phần kết quả.
    expect(screen.queryAllByText(/0912345678/).every((node) => node.tagName === "TEXTAREA")).toBe(true);
  });

  it("lỗi mạng: báo bằng tiếng Việt và cho thử lại", async () => {
    let attempt = 0;
    const { text, response } = analysis("g02");
    stubApi({
      "GET /api/config": () => json(config),
      "POST /api/analyze": () => {
        attempt += 1;
        if (attempt === 1) throw new TypeError("fetch failed");
        return json(response);
      },
      "POST /api/explain": () => json(explanation("g02").response),
    });
    render(<AnalyzePage />);
    const user = await submit(text);
    expect(await screen.findByRole("alert")).toHaveTextContent(/Không kết nối được máy chủ/);
    await user.click(screen.getByRole("button", { name: /Thử lại/ }));
    await screen.findByRole("heading", { name: "Kết quả" });
    expect(attempt).toBe(2);
  });

  it("chọn câu khác khi câu trước chưa xong: kết quả muộn của câu cũ không ghi đè câu mới", async () => {
    const older = analysis("g02");
    const newer = analysis("g08");
    expect(newer.response.sentiment.label).not.toBe(older.response.sentiment.label);
    const firstReply = deferred<Response>();
    let requests = 0;
    stubApi({
      "GET /api/config": () => json(config),
      "POST /api/analyze": () => {
        requests += 1;
        return requests === 1 ? firstReply.promise : json(newer.response);
      },
      "POST /api/explain": () => json(explanation("g08").response),
    });
    render(<AnalyzePage />);
    const user = await submit(older.text);
    // Câu đầu còn chờ: chọn một câu mẫu để chạy câu mới.
    const sample = (config as { ui_samples: { text: string }[] }).ui_samples[0].text;
    await user.click(await screen.findByRole("button", { name: sample }));
    await screen.findByRole("heading", { name: "Kết quả" });

    firstReply.resolve(json(older.response));
    await waitFor(() => expect(callsTo("POST /api/explain").length).toBeGreaterThan(0));
    // Câu thông báo cho trình đọc màn hình luôn phản ánh kết quả đang hiển thị.
    const summary = screen.getByText(/^Kết quả: cảm xúc/);
    expect(summary).toHaveTextContent(labelMeta("sentiment", newer.response.sentiment.label).vi);
    expect(summary).not.toHaveTextContent(labelMeta("sentiment", older.response.sentiment.label).vi);
  });

  it("so sánh mô hình mặc định tắt; bật thì hỏi riêng và không làm hỏng kết quả chính", async () => {
    const { text, response } = analysis("g02");
    const compare = fixture<{ status: number; body: { baseline?: { name: string } } }>("analyze-g02-compare.json");
    stubApi({
      "GET /api/config": () => json(config),
      "POST /api/analyze": async (request) => {
        const body = (await request.clone().json()) as { compare: boolean };
        return body.compare ? json(compare.body, compare.status) : json(response);
      },
      "POST /api/explain": () => json(explanation("g02").response),
    });
    render(<AnalyzePage />);
    const toggle = await screen.findByRole("switch", { name: /So sánh với mô hình cơ sở/ });
    expect(toggle).toHaveAttribute("aria-checked", "false");

    const user = await submit(text);
    await screen.findByRole("heading", { name: "Kết quả" });
    expect(callsTo("POST /api/analyze").map((call) => (call.body as { compare: boolean }).compare)).toEqual([false]);

    await user.click(toggle);
    await screen.findByRole("region", { name: "So sánh mô hình" });
    if (compare.status === 200) {
      // API đã có mô hình cơ sở: khung so sánh hiện tên mô hình đó.
      expect(await screen.findByText(compare.body.baseline?.name ?? "")).toBeInTheDocument();
    } else {
      // Chưa làm xong (501): báo gọn, kết quả chính vẫn còn.
      await screen.findByText(/Tính năng này chưa sẵn sàng trên máy chủ/);
      expect(screen.getByRole("heading", { name: "Kết quả" })).toBeInTheDocument();
    }
    expect(callsTo("POST /api/analyze").map((call) => (call.body as { compare: boolean }).compare)).toEqual([false, true]);
  });
});
