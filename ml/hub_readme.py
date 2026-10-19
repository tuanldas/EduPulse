"""README (thẻ mô hình) của repo Hugging Face, tiếng Việt, dựng từ manifest.json và model_card.json.

Mọi con số lấy từ thẻ mô hình của chính bundle (với bản phát hành đó là số trên TEST), nên README
và trang "Mô hình" của ứng dụng không thể lệch nhau. Module này thuần văn bản: không đọc kết quả
nào khác, không chạm mạng, không chứa câu của bộ dữ liệu.
"""

from __future__ import annotations

from collections.abc import Sequence

from edupulse.bundle_schema import ChecklistRow, Manifest, ModelCard, TaskMetrics
from edupulse.labels import LABELS_VI, TASKS
from ml.plotting import vn

README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"
TASK_TITLE = {"sentiment": "Cảm xúc", "topic": "Chủ đề"}
KIND_TITLE = {"invariance": "Bất biến", "directional": "Định hướng", "robustness": "Độ bền"}
# Mỗi tệp trong repo Hub một dòng mô tả; render_readme từ chối tệp chưa có dòng ở đây.
FILE_NOTES = {
    MANIFEST_NAME: "Phiên bản, nhãn, giao diện ONNX và sha256 của từng tệp.",
    "model.int8.onnx": "Mô hình PhoBERT đa nhiệm đã lượng tử hoá int8.",
    "tokenizer.json": "Tokenizer của PhoBERT-base-v2 (thư viện `tokenizers`).",
    "calibration.json": "Nhiệt độ, phân vị độ tin cậy và ngưỡng gắn cờ theo ngân sách duyệt "
    "(tính trên validation).",
    "model_card.json": "Thẻ mô hình dạng máy đọc, cùng số liệu với trang này.",
    "insight_background.json": "Đếm n-gram nền của tập train cho phần tổng hợp xu hướng "
    "(không chứa câu gốc).",
    "golden_predictions.json": "Dự đoán của các câu tự viết, để kiểm hồi quy khi đổi mã.",
    "baseline/sentiment_linearsvc.skops": "TF-IDF + LinearSVC cho cảm xúc (skops), để so sánh.",
    "baseline/topic_linearsvc.skops": "TF-IDF + LinearSVC cho chủ đề (skops), để so sánh.",
    README_NAME: "Trang này.",
}
BIBTEX = """\
@inproceedings{nguyen2018uitvsfc,
  title     = {UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis},
  author    = {Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu X. V. and
               Truong, Tham T. H. and Nguyen, Ngan Luu-Thuy},
  booktitle = {Proceedings of the 10th International Conference on Knowledge and
               Systems Engineering (KSE 2018)},
  pages     = {19--24},
  year      = {2018},
  doi       = {10.1109/KSE.2018.8573337}
}

@inproceedings{nguyen2020phobert,
  title     = {PhoBERT: Pre-trained language models for Vietnamese},
  author    = {Nguyen, Dat Quoc and Nguyen, Anh Tuan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages     = {1037--1042},
  year      = {2020},
  doi       = {10.18653/v1/2020.findings-emnlp.92}
}"""


def percent(value: float, digits: int = 1) -> str:
    return vn(value * 100, digits)


def optional_percent(value: float | None) -> str:
    return "-" if value is None else percent(value)


def table(header: Sequence[str], rows: Sequence[Sequence[str]], right_from: int = 1) -> str:
    """Bảng Markdown; các cột từ `right_from` trở đi căn phải (cột số)."""
    align = ["---" if i < right_from else "---:" for i in range(len(header))]
    return "\n".join(
        "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
        for row in (header, align, *rows)
    )


def front_matter(manifest: Manifest) -> str:
    """Siêu dữ liệu YAML của Hub: giấy phép, ngôn ngữ, mô hình tiền huấn luyện, dữ liệu, tác vụ."""
    lines = [
        "---",
        "language:",
        "- vi",
        "license: agpl-3.0",
        f"base_model: {manifest.base_model.name}",
        "datasets:",
        f"- {manifest.dataset.name}",
        "library_name: onnx",
        "pipeline_tag: text-classification",
        "tags:",
        "- vietnamese",
        "- phobert",
        "- sentiment-analysis",
        "- topic-classification",
        "- multi-task-learning",
        "- education",
        "- onnx",
        "- int8",
        "metrics:",
        "- accuracy",
        "- f1",
        "---",
    ]
    return "\n".join(lines)


def header_block(manifest: Manifest, source_url: str) -> str:
    dataset, base = manifest.dataset, manifest.base_model
    sizes = " / ".join(vn(dataset.splits[name], 0) for name in ("train", "validation", "test"))
    rows = [
        [
            "Phiên bản",
            f"`{manifest.model_version}`, bundle `{manifest.bundle_version}` (phát hành)",
        ],
        [
            "Dựng",
            f"{manifest.created_at:%d/%m/%Y} từ commit `{manifest.git_commit}` của mã nguồn",
        ],
        [
            "Mô hình tiền huấn luyện",
            f"[{base.name}](https://huggingface.co/{base.name}) (giấy phép {base.license}), "
            f"revision `{base.revision[:12]}`",
        ],
        ["Kiến trúc", "Một bộ mã hoá PhoBERT dùng chung và hai đầu phân loại (đa nhiệm)"],
        ["Định dạng", "ONNX, lượng tử hoá int8, chạy trên CPU bằng ONNX Runtime"],
        [
            "Dữ liệu",
            f"[{dataset.name}](https://huggingface.co/datasets/{dataset.name}), "
            f"{vn(sum(dataset.splits.values()), 0)} câu, chia train / validation / test = {sizes}",
        ],
    ]
    if source_url:
        rows.append(["Mã nguồn", source_url])
    return "\n".join(
        [
            "# EduPulse · PhoBERT đa nhiệm cho góp ý của sinh viên (ONNX int8)",
            "",
            "Mô hình đọc một câu góp ý tiếng Việt của sinh viên và trả về cùng lúc **cảm xúc** "
            "(tiêu cực, trung tính, tích cực) và **chủ đề** (giảng viên, chương trình, cơ sở vật "
            "chất, khác). Đây là mô hình của đồ án học thuật EduPulse, không phải sản phẩm "
            "thương mại.",
            "",
            table(["", ""], rows, right_from=2),
        ]
    )


def use_block(card: ModelCard) -> str:
    default = next(row.budget for row in card.review_budget if row.default)
    return "\n".join(
        [
            "## Mục đích sử dụng",
            "",
            card.intended_use,
            "",
            f"**Không dùng cho:** {card.out_of_scope}",
            "",
            "Dự đoán chỉ là gợi ý cho người đọc, không phải quyết định. EduPulse gắn cờ các câu có "
            f"độ tin cậy thấp để người duyệt; ngân sách duyệt mặc định là {default}% số câu.",
        ]
    )


def overall_row(task: str, metrics: TaskMetrics) -> list[str]:
    return [
        TASK_TITLE[task],
        percent(metrics.accuracy),
        percent(metrics.macro_f1),
        percent(metrics.weighted_f1),
        vn(metrics.ece, 3),
    ]


def per_class_rows(card: ModelCard) -> list[list[str]]:
    rows = []
    for task in TASKS:
        for item in getattr(card.metrics, task).per_class:
            rows.append(
                [
                    TASK_TITLE[task],
                    LABELS_VI[task].get(item.label, item.label),
                    percent(item.precision),
                    percent(item.recall),
                    percent(item.f1),
                    vn(item.support, 0),
                ]
            )
    return rows


def baseline_table(card: ModelCard) -> str:
    rows = [
        [
            f"**{row.name}**" if row.deployed else row.name,
            percent(row.sentiment_accuracy),
            percent(row.sentiment_macro_f1),
            percent(row.topic_accuracy),
            percent(row.topic_macro_f1),
            "-" if row.sentiment_ece is None else vn(row.sentiment_ece, 3),
        ]
        for row in card.baselines
    ]
    header = [
        "Mô hình",
        "Cảm xúc · Accuracy",
        "Cảm xúc · Macro-F1",
        "Chủ đề · Accuracy",
        "Chủ đề · Macro-F1",
        "ECE cảm xúc",
    ]
    return table(header, rows)


def architecture_table(card: ModelCard) -> str:
    rows = []
    for arch in card.architectures:
        cells = []
        for stat in (arch.sentiment_macro_f1, arch.topic_macro_f1):
            text = percent(stat.mean)
            cells.append(text if stat.n == 1 else f"{text} ± {percent(stat.std)}")
        rows.append([arch.name, *cells])
    return table(["Kiến trúc", "Cảm xúc · Macro-F1", "Chủ đề · Macro-F1"], rows)


def results_block(card: ModelCard) -> str:
    overall = [overall_row(task, getattr(card.metrics, task)) for task in TASKS]
    lines = [
        f"## Kết quả trên tập test ({vn(card.metrics.n, 0)} câu)",
        "",
        "Số liệu của bản int8 trong repo này, chấm trên tập test chuẩn của UIT-VSFC; tập này không "
        "dùng để chọn mô hình, epoch hay ngưỡng. Đơn vị là %, riêng ECE (độ lệch hiệu chỉnh, càng "
        "nhỏ càng tốt) là số thập phân. Macro-F1 là chỉ số chính vì Trung tính (cảm xúc) và Khác "
        "(chủ đề) là các lớp nhỏ.",
        "",
        table(["Tác vụ", "Accuracy", "Macro-F1", "Weighted-F1", "ECE"], overall),
        "",
        "### Từng lớp",
        "",
        table(
            ["Tác vụ", "Lớp", "Precision", "Recall", "F1", "Số câu"],
            per_class_rows(card),
            right_from=2,
        ),
    ]
    if card.baselines:
        lines += ["", "### So sánh với các mô hình cơ sở (cùng tập test)", "", baseline_table(card)]
    if card.architectures:
        lines += [
            "",
            "### Đa nhiệm so với đơn nhiệm",
            "",
            "Chỉ là thí nghiệm sơ bộ của nghiên cứu (không phải số test, không phải bản triển "
            "khai):",
            "",
            architecture_table(card),
        ]
    return "\n".join(lines)


def budget_block(card: ModelCard) -> str:
    rows = [
        [
            f"{row.budget}%" + (" (mặc định)" if row.default else ""),
            percent(row.efficiency.flagged_share),
            percent(row.efficiency.caught_sentiment_errors),
            percent(row.efficiency.caught_topic_errors),
            percent(row.efficiency.accept_accuracy_both),
            percent(row.efficiency.all_accuracy_both),
        ]
        for row in card.review_budget
    ]
    header = [
        "Ngân sách",
        "Câu gắn cờ",
        "Lỗi cảm xúc bắt được",
        "Lỗi chủ đề bắt được",
        "Đúng cả hai nhãn, phần không cờ",
        "Đúng cả hai nhãn, toàn bộ",
    ]
    return "\n".join(
        [
            "### Ngân sách duyệt (trên test)",
            "",
            "Các câu có độ tin cậy thấp nhất được gắn cờ cho người duyệt. Bảng cho biết mỗi mức "
            "ngân sách bắt được bao nhiêu lỗi và độ chính xác (đúng cả hai nhãn) của phần "
            "không cờ.",
            "",
            table(header, rows),
        ]
    )


def checklist_block(rows: Sequence[ChecklistRow]) -> str:
    cells = [
        [row.name, KIND_TITLE[row.kind], vn(row.n, 0), optional_percent(row.pass_rate)]
        for row in rows
    ]
    return "\n".join(
        [
            "### Kiểm thử hành vi (CheckList, dựng từ câu validation)",
            "",
            table(["Kiểm thử", "Loại", "Số câu", "Đạt (%)"], cells, right_from=2),
        ]
    )


def limitations_block(card: ModelCard) -> str:
    lines = ["## Giới hạn đã biết", ""]
    lines += [f"- {item}" for item in card.limitations]
    lines.append(f"- {card.notes.x86}")
    if card.notes.demo:
        lines.append(f"- {card.notes.demo}")
    return "\n".join(lines)


def ethics_block(manifest: Manifest) -> str:
    total = vn(sum(manifest.dataset.splits.values()), 0)
    return "\n".join(
        [
            "## Dữ liệu, quyền riêng tư và đạo đức",
            "",
            f"- **Dữ liệu huấn luyện:** UIT-VSFC, {total} câu góp ý của sinh viên Trường Đại học "
            "Công nghệ Thông tin (ĐHQG-HCM), giai đoạn 2013-2016. Bộ dữ liệu chỉ dành cho học "
            "thuật, phi thương mại.",
            "- **Repo này không chứa câu gốc của UIT-VSFC.** Trước khi đẩy, mọi tệp văn bản được "
            "đối chiếu với toàn bộ câu dài của bộ dữ liệu.",
            "- **Không dùng để đánh giá cá nhân.** Mô hình phân tích góp ý theo nhóm câu, không "
            "dùng để chấm điểm, xếp loại hay kỷ luật giảng viên hoặc sinh viên.",
            "- **Con người quyết định.** Câu có độ tin cậy thấp được gắn cờ để người duyệt; hệ "
            "thống không tự ra quyết định.",
            "- **Thông tin cá nhân.** Ứng dụng thay số điện thoại, email, mã số sinh viên và tên "
            "người (cố gắng tối đa, tên viết thường có thể sót) trước khi phân tích.",
        ]
    )


def usage_snippet(manifest: Manifest) -> str:
    spec = manifest.onnx
    names = ", ".join(spec.output_names)
    quoted = ", ".join(f'"{name}"' for name in spec.output_names)
    feed = f'{{"{spec.input_names[0]}": ids, "{spec.input_names[1]}": np.ones_like(ids)}}'
    return "\n".join(
        [
            "import numpy as np",
            "import onnxruntime as ort",
            "from tokenizers import Tokenizer",
            "",
            'tokenizer = Tokenizer.from_file("tokenizer.json")',
            f"tokenizer.enable_truncation(max_length={manifest.max_len})",
            'session = ort.InferenceSession("model.int8.onnx", providers=["CPUExecutionProvider"])',
            "",
            "# Câu phải được chuẩn hoá và tách từ (VnCoreNLP RDRSegmenter, nối các tiếng của một",
            '# từ bằng "_") như lúc huấn luyện.',
            'text = "giảng_viên dạy rất nhiệt_tình"',
            "ids = np.array([tokenizer.encode(text).ids], dtype=np.int64)",
            f"{names} = session.run([{quoted}], {feed})",
        ]
    )


def usage_block(manifest: Manifest, repo_id: str) -> str:
    spec = manifest.onnx
    labels = "; ".join(
        f"{TASK_TITLE[task].lower()}: {', '.join(manifest.labels[task])}" for task in TASKS
    )
    return "\n".join(
        [
            "## Cách dùng",
            "",
            "Ứng dụng EduPulse tải bundle này bằng `make fetch-artifacts`: đặt "
            f"`EDUPULSE_HF_REPO={repo_id}` và `EDUPULSE_HF_REVISION` (commit sha đầy đủ của repo "
            "này) trong `.env`; sha256 từng tệp được kiểm theo `manifest.json`. Có thể chạy trực "
            "tiếp bằng ONNX Runtime:",
            "",
            "```python",
            usage_snippet(manifest),
            "```",
            "",
            f"- Đầu vào: `{spec.input_names[0]}` và `{spec.input_names[1]}` (int64, tối đa "
            f"{manifest.max_len} token, id đệm {spec.pad_id}); phiên bản chuẩn hoá văn bản "
            f"`{manifest.normalizer_version}`.",
            f"- Đầu ra: `{spec.output_names[0]}` và `{spec.output_names[1]}`. Thứ tự nhãn là "
            f"{labels}.",
            "- Xác suất đã hiệu chỉnh bằng softmax(logits / T); T của từng đầu nằm trong "
            "`calibration.json`.",
            f"- Lượng tử hoá: {spec.quantization}.",
        ]
    )


def files_block(manifest: Manifest) -> str:
    shipped = {MANIFEST_NAME, README_NAME, *manifest.files}
    undocumented = sorted(shipped - FILE_NOTES.keys())
    if undocumented:
        raise ValueError(f"README chưa mô tả các tệp: {undocumented}")
    rows = [[f"`{name}`", note] for name, note in FILE_NOTES.items() if name in shipped]
    digests = "\n".join(f"{digest}  {name}" for name, digest in manifest.files.items())
    return "\n".join(
        [
            "## Tệp trong repo",
            "",
            table(["Tệp", "Nội dung"], rows, right_from=2),
            "",
            "Kiểm toàn vẹn (`sha256sum -c` hoặc `shasum -a 256 -c`, chạy trong thư mục đã tải):",
            "",
            "```",
            digests,
            "```",
        ]
    )


def license_block(manifest: Manifest, source_url: str) -> str:
    code = f"AGPL-3.0, tại {source_url}" if source_url else "AGPL-3.0"
    return "\n".join(
        [
            "## Giấy phép",
            "",
            "- Trọng số và các tệp trong repo này: **AGPL-3.0**, kế thừa từ "
            f"{manifest.base_model.name} (VinAI, {manifest.base_model.license}).",
            f"- Mã nguồn EduPulse: {code}.",
            "- Tách từ: VnCoreNLP (RDRSegmenter), GPL-3.0+, dùng ở phía ứng dụng và không nằm "
            "trong repo này.",
            "- Dữ liệu UIT-VSFC: chỉ dùng cho học thuật, phi thương mại; khi dùng mô hình này, "
            "hãy tuân theo điều khoản của bộ dữ liệu.",
        ]
    )


def citation_block() -> str:
    return "\n".join(
        [
            "## Trích dẫn",
            "",
            "Nếu dùng mô hình này, vui lòng trích dẫn bộ dữ liệu UIT-VSFC và PhoBERT:",
            "",
            "```bibtex",
            BIBTEX,
            "```",
        ]
    )


def render_readme(
    manifest: Manifest, card: ModelCard, *, repo_id: str, source_url: str = ""
) -> str:
    """README của repo Hub; `source_url` rỗng thì không có dòng liên kết mã nguồn."""
    blocks = [
        front_matter(manifest),
        header_block(manifest, source_url),
        use_block(card),
        results_block(card),
        budget_block(card),
    ]
    if card.checklist:
        blocks.append(checklist_block(card.checklist))
    blocks += [
        limitations_block(card),
        ethics_block(manifest),
        usage_block(manifest, repo_id),
        files_block(manifest),
        license_block(manifest, source_url),
        citation_block(),
    ]
    return "\n\n".join(blocks) + "\n"
