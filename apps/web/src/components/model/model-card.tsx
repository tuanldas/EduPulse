import * as React from "react";

import { Icon, type IconName } from "@/components/shared/icon";
import { SectionHeading } from "@/components/shared/section-heading";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { ModelCard } from "@/lib/api/types";
import { formatNumber, formatPercent } from "@/lib/format";

type ChecklistRow = ModelCard["checklist"][number];

const CHECK_KIND_VI: Record<ChecklistRow["kind"], string> = {
  invariance: "bất biến",
  directional: "đổi chiều",
  robustness: "bền vững",
};

function Row({ icon, title, children }: { icon: IconName; title: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="flex items-start gap-2 border-t border-divider py-3 pr-4 font-semibold first:border-t-0 max-[640px]:pb-0">
        <Icon name={icon} size="sm" className="mt-0.5 text-muted-foreground" />
        {title}
      </dt>
      <dd className="m-0 border-t border-divider py-3 first-of-type:border-t-0 max-[640px]:border-t-0 max-[640px]:pt-1">
        {children}
      </dd>
    </>
  );
}

function ChecklistList({ rows }: { rows: ChecklistRow[] }) {
  return (
    <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
      {rows.map((row) => (
        <li key={row.name} className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
          <b className="font-semibold">{row.name}</b>
          <Badge variant="gray">{CHECK_KIND_VI[row.kind]}</Badge>
          <span className="text-muted-foreground tabular-nums">
            {row.pass_rate == null ? "chưa có tỉ lệ đạt" : `đạt ${formatPercent(row.pass_rate, 1)}`} · {formatNumber(row.n)} câu
          </span>
          {row.note ? <span className="text-sm text-muted-foreground">— {row.note}</span> : null}
        </li>
      ))}
    </ul>
  );
}

/**
 * Mục "Model card": mục đích, giới hạn, dữ liệu, mô hình, kiểm tra hành vi, phiên bản. Nội dung do model
 * card trong bundle quyết định (mục đích, không dùng cho, giới hạn, ghi chú, kết quả CheckList); phần
 * mô tả kiến trúc là chữ cố định vì không đổi theo bản.
 */
export function ModelCardSection({ card }: { card: ModelCard }) {
  const splitName = card.metrics.split === "test" ? "test" : "validation";
  return (
    <section aria-labelledby="model-card" className="flex flex-col gap-4">
      <SectionHeading id="model-card" icon="badge" className="mt-4">
        Model card
      </SectionHeading>
      <Card>
        <dl className="grid grid-cols-[200px_minmax(0,1fr)] text-[15px] leading-[1.55] max-[640px]:grid-cols-1">
          <Row icon="flag" title="Mục đích">
            {card.intended_use}
          </Row>
          <Row icon="block" title="Không dùng cho">
            {card.out_of_scope}
          </Row>
          <Row icon="dataset" title="Dữ liệu">
            UIT-VSFC: góp ý của sinh viên Trường ĐH Công nghệ Thông tin, ĐHQG-HCM (2013–2016), chia train / validation /
            test theo chuẩn của bộ dữ liệu. Tập đánh giá ở trên là tập {splitName} ({formatNumber(card.metrics.n)} câu).
          </Row>
          <Row icon="model_training" title="Mô hình">
            PhoBERT-base-v2 (VinAI): 1 bộ mã hoá + 2 đầu ra (cảm xúc, chủ đề); tách từ RDRSegmenter (VnCoreNLP); lượng tử
            hoá int8 chạy bằng ONNX Runtime. Xác suất hiệu chỉnh bằng temperature scaling. Giải thích bằng cách bỏ từng từ
            (occlusion).
          </Row>
          <Row icon="warning" title="Giới hạn đã biết">
            <ul className="m-0 list-disc pl-5">
              {card.limitations.map((item) => (
                <li key={item} className="my-0.5">
                  {item}
                </li>
              ))}
            </ul>
          </Row>
          <Row icon="fact_check" title="Kiểm tra hành vi">
            Đổi “thầy” ↔ “cô”, đổi tên, bỏ dấu, chèn thông tin cá nhân, thêm “không” để đảo nghĩa (CheckList).
            <div className="mt-2">
              {card.checklist.length > 0 ? (
                <ChecklistList rows={card.checklist} />
              ) : (
                <span className="text-sm text-muted-foreground">
                  Chưa có kết quả kiểm tra cho bản này; kết quả cập nhật sau khi chốt mô hình.
                </span>
              )}
            </div>
          </Row>
          <Row icon="tag" title="Phiên bản">
            <code>
              {card.model_version} · {card.bundle_version}
            </code>{" "}
            <Badge variant={card.stage === "release" ? "default" : "gray"}>
              {card.stage === "release" ? "bản chính thức" : "bản smoke"}
            </Badge>
          </Row>
          <Row icon="info" title="Ghi chú triển khai">
            <ul className="m-0 list-disc pl-5">
              <li className="my-0.5">{card.notes.x86}</li>
              {card.notes.demo ? <li className="my-0.5">{card.notes.demo}</li> : null}
            </ul>
          </Row>
        </dl>
      </Card>
    </section>
  );
}
