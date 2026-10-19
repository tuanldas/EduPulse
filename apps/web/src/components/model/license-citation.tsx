"use client";

import { toast } from "sonner";

import { Icon } from "@/components/shared/icon";
import { Caption, SectionHeading } from "@/components/shared/section-heading";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useAppConfig } from "@/lib/app-config";

const LICENSES: readonly { name: string; license: string }[] = [
  { name: "Mã nguồn EduPulse", license: "AGPL-3.0" },
  { name: "PhoBERT-base-v2 (VinAI)", license: "AGPL-3.0" },
  { name: "VnCoreNLP · RDRSegmenter", license: "GPL-3.0+" },
  { name: "Dữ liệu UIT-VSFC", license: "Học thuật, phi thương mại" },
];

// Hai bài gốc, đã đối chiếu với trang của nhà xuất bản (IEEE Xplore, ACL Anthology).
const BIBTEX = `@inproceedings{nguyen2018uitvsfc,
  title     = {UIT-VSFC: Vietnamese Students'
               Feedback Corpus for Sentiment
               Analysis},
  author    = {Nguyen, Kiet Van and
               Nguyen, Vu Duc and
               Nguyen, Phu X. V. and
               Truong, Tham T. H. and
               Nguyen, Ngan Luu-Thuy},
  booktitle = {2018 10th International
               Conference on Knowledge and
               Systems Engineering (KSE)},
  pages     = {19--24},
  year      = {2018},
  doi       = {10.1109/KSE.2018.8573337}
}

@inproceedings{nguyen2020phobert,
  title     = {{PhoBERT}: Pre-trained language
               models for {V}ietnamese},
  author    = {Nguyen, Dat Quoc and
               Nguyen, Anh Tuan},
  booktitle = {Findings of the Association for
               Computational Linguistics:
               EMNLP 2020},
  pages     = {1037--1042},
  year      = {2020},
  doi       = {10.18653/v1/2020.findings-emnlp.92}
}`;

async function copyCitation() {
  try {
    await navigator.clipboard.writeText(BIBTEX);
    toast.success("Đã sao chép trích dẫn BibTeX.");
  } catch {
    toast.error("Không sao chép được. Hãy chọn và sao chép thủ công.");
  }
}

function isGithub(url: string): boolean {
  try {
    return new URL(url).hostname === "github.com";
  } catch {
    return false;
  }
}

/** Mục "Giấy phép, mã nguồn & trích dẫn": bảng giấy phép, liên kết mã nguồn từ `/api/config` và BibTeX có nút sao chép. */
export function LicenseCitation() {
  const state = useAppConfig();
  const sourceUrl = state.status === "ready" ? state.config.source_url : "";
  return (
    <section aria-labelledby="giay-phep" className="flex flex-col gap-4">
      <SectionHeading id="giay-phep" icon="gavel" className="mt-4">
        Giấy phép, mã nguồn &amp; trích dẫn
      </SectionHeading>
      <div className="grid grid-cols-2 items-start gap-4 max-[900px]:grid-cols-1">
        <Card>
          <dl className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-2.5 text-[15px]">
            {LICENSES.map((item) => (
              <div key={item.name} className="contents">
                <dt>{item.name}</dt>
                <dd className="m-0 justify-self-end text-right">
                  <Badge variant="gray">{item.license}</Badge>
                </dd>
              </div>
            ))}
          </dl>
          {sourceUrl ? (
            <div className="flex flex-wrap gap-2">
              <a href={sourceUrl} target="_blank" rel="noopener noreferrer" className={buttonVariants({ variant: "outline" })}>
                <Icon name="code" size="sm" />
                {isGithub(sourceUrl) ? "Mã nguồn trên GitHub" : "Mã nguồn"}
              </a>
            </div>
          ) : null}
          <Caption>
            {sourceUrl ? <code>{sourceUrl}</code> : "Liên kết mã nguồn chưa được cấu hình trên máy chủ này."}
            <br />
            Vì PhoBERT-base-v2 dùng AGPL-3.0, toàn bộ mã nguồn ứng dụng được công khai tại liên kết trên.
          </Caption>
        </Card>
        <div className="flex min-w-0 flex-col gap-2">
          <div className="text-[15px] font-semibold">Trích dẫn</div>
          <div className="relative">
            <pre
              tabIndex={0}
              aria-label="Trích dẫn BibTeX"
              className="overflow-auto rounded-lg bg-muted p-3.5 pr-12 font-mono text-[13px] leading-[1.6] break-words whitespace-pre-wrap outline-none focus-visible:ring-3 focus-visible:ring-ring/70"
            >
              {BIBTEX}
            </pre>
            <Button
              variant="ghost"
              size="icon-sm"
              className="absolute top-2 right-2 text-muted-foreground"
              aria-label="Sao chép trích dẫn BibTeX"
              onClick={() => void copyCitation()}
            >
              <Icon name="content_copy" size="sm" />
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
