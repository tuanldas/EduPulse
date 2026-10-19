import * as React from "react";

import { Icon, type IconName } from "@/components/shared/icon";
import { PiiChip } from "@/components/shared/pii-chip";
import { Caption, SectionHeading } from "@/components/shared/section-heading";
import { Card } from "@/components/ui/card";
import { PII_TAG } from "@/lib/labels";

function Block({ icon, title, children }: { icon: IconName; title: string; children: React.ReactNode }) {
  return (
    <Card className="gap-2">
      <span className="flex size-9 items-center justify-center rounded-lg bg-primary-soft text-primary-text">
        <Icon name={icon} />
      </span>
      <h3 className="text-base font-semibold">{title}</h3>
      <p className="text-sm leading-[1.55] text-muted-foreground">{children}</p>
    </Card>
  );
}

/** Mục "Quyền riêng tư & vai trò của con người": ba khối nguyên tắc và dòng tự đánh giá rủi ro. */
export function PrivacyBlocks() {
  return (
    <section aria-labelledby="rieng-tu" className="flex flex-col gap-4">
      <SectionHeading id="rieng-tu" icon="lock" className="mt-4">
        Quyền riêng tư &amp; vai trò của con người
      </SectionHeading>
      <div className="grid grid-cols-3 gap-4 max-[900px]:grid-cols-1">
        <Block icon="cloud_off" title="Không lưu dữ liệu">
          Câu và file bạn tải lên chỉ nằm trong bộ nhớ của phiên làm việc và mất khi đóng tab. Không ghi log nội dung.
        </Block>
        <Block icon="visibility_off" title="Tự ẩn thông tin cá nhân">
          SĐT, email, MSSV, CCCD, URL được thay bằng <PiiChip>{PII_TAG.phone}</PiiChip> <PiiChip>{PII_TAG.email}</PiiChip>{" "}
          <PiiChip>{PII_TAG.student_id}</PiiChip>… trước khi phân tích và xuất file. Tên người{" "}
          <PiiChip>{PII_TAG.name}</PiiChip> chỉ ở mức cố gắng tối đa (best effort): tên viết thường, biệt danh có thể sót →
          hãy ẩn danh trước khi tải file.
        </Block>
        <Block icon="person_check" title="Con người quyết định">
          Câu có độ tin cậy thấp được gắn cờ để người duyệt; nhãn đã sửa có thể xuất ra để huấn luyện lại. Hệ thống không
          tự ra quyết định.
        </Block>
      </div>
      <Caption>
        Tự đánh giá mức rủi ro: <b className="font-semibold text-foreground">thấp</b> — phân tích góp ý về giảng dạy, không
        chấm điểm hay giám sát người học (giải trình trong báo cáo).
      </Caption>
    </section>
  );
}
