import { AnalyzePage } from "@/components/analyze/analyze-page";

// `title.template` của layout gốc chỉ áp cho các trang con, nên trang gốc ghi đủ tên.
export const metadata = { title: { absolute: "Phân tích câu · EduPulse" } };

export default function Page() {
  return <AnalyzePage />;
}
