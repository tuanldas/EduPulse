import { Suspense } from "react";

import { DashboardSkeleton, DashboardWorkspace } from "@/components/dashboard/dashboard-workspace";
import { PageHeader } from "@/components/shell/page-header";

export const metadata = { title: "Dashboard & insight" };

export default function Page() {
  return (
    <>
      <PageHeader
        icon="monitoring"
        title="Dashboard & insight"
        description="Cảm xúc theo chủ đề, so với mức nền của từng chủ đề, kèm từ khoá, câu tiêu biểu và khuyến nghị để cán bộ đọc nhanh."
      />
      {/* `useSearchParams` (đọc ?batch=) cần Suspense để trang export tĩnh dựng được. */}
      <Suspense fallback={<DashboardSkeleton />}>
        <DashboardWorkspace />
      </Suspense>
    </>
  );
}
