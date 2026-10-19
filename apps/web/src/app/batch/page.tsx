import { BatchWorkspace } from "@/components/batch/batch-workspace";
import { PageHeader } from "@/components/shell/page-header";

export const metadata = { title: "Phân tích hàng loạt" };

export default function Page() {
  return (
    <>
      <PageHeader
        icon="upload_file"
        title="Phân tích hàng loạt"
        description="Tải file góp ý, chọn cột văn bản rồi chạy. Sau đó duyệt các câu được gắn cờ, sửa nhãn nếu cần và tải kết quả về."
      />
      <BatchWorkspace />
    </>
  );
}
