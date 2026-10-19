import { Callout } from "./callout";

interface DemoUnavailableNoticeProps {
  /** `demo_reason` của `/api/config`, nếu máy chủ có ghi. */
  reason?: string | null;
  className?: string;
}

/** Thông báo khi dữ liệu demo (câu mẫu, file mẫu, batch demo) không có vì bản build không tải được dataset. */
export function DemoUnavailableNotice({ reason, className }: DemoUnavailableNoticeProps) {
  return (
    <Callout tone="info" icon="cloud_off" title="Dữ liệu demo chưa sẵn" role="status" className={className}>
      (bản build không tải được dataset).{reason ? ` ${reason}` : ""} Bạn vẫn dùng được các chức năng khác bằng dữ liệu của mình.
    </Callout>
  );
}
