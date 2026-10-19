import Link from "next/link";

import { Icon } from "@/components/shared/icon";
import { buttonVariants } from "@/components/ui/button";
import { PageHeader } from "@/components/shell/page-header";

export const metadata = { title: { absolute: "Không tìm thấy trang · EduPulse" } };

// Next sinh `404.html` từ file này; FastAPI trả nó khi trình duyệt mở một đường dẫn không có.
export default function NotFound() {
  return (
    <>
      <PageHeader
        icon="search"
        title="Không tìm thấy trang"
        description="Đường dẫn này không có trong EduPulse. Hãy chọn một trang ở thanh điều hướng."
      />
      <div>
        <Link href="/" className={buttonVariants({ variant: "default" })}>
          <Icon name="rate_review" size="sm" />
          Về trang Phân tích câu
        </Link>
      </div>
    </>
  );
}
