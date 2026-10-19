// File góp ý tự viết cho các spec E2E của trang 2 và 3: không có câu nào lấy từ UIT-VSFC. Mỗi câu được ghép từ
// các vế do người viết soạn (người/đối tượng + ý chính + vế kết) theo một bảng số lượng cố định, nên lần nào
// chạy cũng ra đúng cùng một file. Bảng số lượng đặt sao cho nhóm B bị chê về giảng viên nhiều hơn hẳn mức nền,
// góp ý về chương trình gần như toàn tiêu cực và chủ đề cơ sở vật chất xuất hiện dày, để trang 3 có đủ ba kiểu
// cảnh báo thật để xem; nhãn cuối cùng vẫn do mô hình thật quyết, không phải do bảng này.
//
// Tệp này không import gì (chỉ cú pháp TypeScript xoá được) để script dò tạm chạy thẳng bằng Node.

type FeedbackTopic = "lecturer" | "training_program" | "facility" | "other";
type Tone = "negative" | "neutral" | "positive";

interface TopicPhrases {
  /** Người hoặc đối tượng đứng đầu câu; `[""]` nghĩa là mỗi vế ý đã là một câu hoàn chỉnh. */
  subjects: readonly string[];
  bodies: Record<Tone, readonly string[]>;
  closers: Record<Tone, readonly string[]>;
}

const PHRASES: Record<FeedbackTopic, TopicPhrases> = {
  lecturer: {
    subjects: [
      "Thầy",
      "Cô",
      "Giảng viên môn Cơ sở dữ liệu",
      "Thầy dạy môn Lập trình web",
      "Cô dạy môn Mạng máy tính",
      "Giảng viên môn Kinh tế vi mô",
      "Cô dạy môn Toán rời rạc",
      "Thầy dạy môn Pháp luật đại cương",
      "Giảng viên môn Tiếng Anh chuyên ngành",
      "Giảng viên môn Hệ điều hành",
      "Cô dạy môn Marketing căn bản",
      "Thầy dạy môn Kế toán cơ bản",
    ],
    bodies: {
      negative: [
        "giảng quá nhanh nên em không theo kịp",
        "ít khi trả lời câu hỏi của sinh viên trong giờ học",
        "hay đến lớp muộn và kết thúc buổi học sớm",
        "chỉ đọc lại slide mà không giải thích thêm",
        "chấm bài chậm và không nhận xét lỗi sai",
        "ra nhiều bài tập nhưng không hướng dẫn cách làm",
        "thiếu nhiệt tình khi sinh viên hỏi bài",
        "dạy khó hiểu, ví dụ ít và không gắn với thực tế",
        "nghỉ nhiều mà không dạy bù nên lớp bị hụt kiến thức",
        "nói nhỏ và đều đều khiến cả lớp buồn ngủ",
        "cho nghỉ đột xuất mà không báo trước",
        "chấm điểm thiếu công bằng giữa các nhóm",
        "thường lạc đề sang chuyện cá nhân trong giờ học",
        "không cập nhật bài giảng nên nhiều ví dụ đã lỗi thời",
        "hay ngắt lời khi sinh viên đang phát biểu",
        "đặt câu hỏi quá khó mà không gợi ý hướng giải",
        "ít tương tác với lớp nên buổi học rất nặng nề",
        "trả lời email của sinh viên rất chậm",
      ],
      neutral: [
        "dạy vào chiều thứ hai và sáng thứ năm",
        "gửi tài liệu qua email trước mỗi buổi học",
        "yêu cầu sinh viên chia nhóm bốn người để làm bài tập",
        "điểm danh vào đầu giờ mỗi buổi",
        "công bố đề cương môn học ở buổi đầu tiên",
        "hẹn gặp sinh viên ở phòng bộ môn vào giờ nghỉ trưa",
      ],
      positive: [
        "giảng bài rất dễ hiểu và có nhiều ví dụ thực tế",
        "sẵn sàng giảng lại khi cả lớp chưa nắm được bài",
        "chuẩn bị bài giảng kỹ và truyền đạt rất rõ ràng",
        "thân thiện và lắng nghe ý kiến của cả lớp",
        "chấm bài nhanh và nhận xét chi tiết từng lỗi sai",
        "tạo không khí lớp học vui vẻ nên em tập trung hơn",
        "hướng dẫn tận tình khi em làm đồ án",
        "dạy đúng giờ và luôn dạy bù đầy đủ khi nghỉ",
        "giải thích chậm rãi nên người học yếu cũng theo được",
        "sẵn sàng trao đổi thêm với sinh viên sau mỗi buổi học",
      ],
    },
    closers: {
      negative: [""],
      neutral: ["", "", " theo thông báo đầu kỳ"],
      positive: ["", ", em rất hài lòng", " nên cả lớp đều thích học", " ở hầu hết các buổi"],
    },
  },
  training_program: {
    subjects: [
      "Môn Cơ sở dữ liệu",
      "Môn Lập trình web",
      "Môn Mạng máy tính",
      "Môn Kinh tế vi mô",
      "Môn Toán rời rạc",
      "Môn Pháp luật đại cương",
      "Môn Tiếng Anh chuyên ngành",
      "Môn Hệ điều hành",
      "Môn Marketing căn bản",
      "Môn Kế toán cơ bản",
      "Chương trình học của ngành",
    ],
    bodies: {
      negative: [
        "nặng lý thuyết và có rất ít bài thực hành",
        "dùng giáo trình đã cũ so với thực tế hiện nay",
        "dồn quá nhiều bài tập và đồ án vào cuối kỳ",
        "có lịch học dày, nhiều buổi khó nằm trong cùng một ngày",
        "có khối lượng kiến thức quá nhiều so với số tiết học",
        "chưa cập nhật công nghệ mới mà doanh nghiệp đang dùng",
        "có đề thi cuối kỳ không sát với nội dung đã học",
        "trùng lặp nội dung với môn của học kỳ trước",
        "thiếu tài liệu tham khảo bằng tiếng Việt",
        "có quá nhiều bài kiểm tra nhỏ nên áp lực kéo dài suốt kỳ",
        "được phân bổ tín chỉ chưa hợp lý so với độ khó",
        "thiếu bài thực hành để áp dụng vào công việc",
        "chia điểm đồ án nhóm chưa rõ ràng",
        "đòi hỏi kiến thức nền mà các môn trước chưa dạy",
      ],
      neutral: [
        "có hai bài kiểm tra giữa kỳ và một bài thi cuối kỳ",
        "gồm ba tín chỉ và kéo dài mười lăm tuần",
        "chia thành phần lý thuyết và phần thực hành",
        "được mở ở học kỳ hai của năm thứ hai",
      ],
      positive: [
        "có nhiều bài thực hành sát với yêu cầu của doanh nghiệp",
        "được sắp xếp hợp lý từ cơ bản đến nâng cao",
        "có giáo trình rõ ràng nên em dễ tự học",
        "giúp em biết cách áp dụng kiến thức vào công việc",
        "có đồ án nhóm rất bổ ích",
      ],
    },
    closers: {
      negative: [""],
      neutral: ["", "", " theo đề cương"],
      positive: ["", ", em thấy rất hữu ích", " nên em học có hứng thú hơn"],
    },
  },
  facility: {
    subjects: [""],
    bodies: {
      negative: [
        "Phòng học nóng và quạt trần kêu to nên rất khó tập trung",
        "Máy chiếu mờ, chữ trên màn hình khó đọc ở dãy bàn cuối",
        "Wifi của trường yếu và hay mất kết nối khi cả lớp cùng dùng",
        "Phòng máy có nhiều máy chạy chậm và hay bị treo",
        "Bàn ghế trong phòng học đã cũ và nhiều chiếc bị gãy",
        "Micro hay bị rè nên em ngồi cuối lớp không nghe rõ",
        "Ổ cắm điện ở giảng đường quá ít để sinh viên sạc máy tính",
        "Căng tin đông và giá đồ ăn khá đắt",
        "Nhà vệ sinh tầng một thường xuyên không có nước",
        "Bãi giữ xe chật nên em phải gửi xe ở ngoài",
        "Thang máy chờ rất lâu vào giờ cao điểm",
        "Phòng thí nghiệm thiếu thiết bị nên các nhóm phải dùng chung",
        "Điều hòa trong giảng đường hỏng nhiều tuần mà chưa ai sửa",
        "Bảng trắng viết không rõ và thiếu bút lông",
        "Thư viện thiếu chỗ ngồi vào mùa thi",
        "Máy in ở khu tự học thường xuyên hết giấy",
      ],
      neutral: [
        "Phòng thực hành nằm ở tầng ba của tòa nhà B",
        "Thư viện mở cửa từ bảy giờ sáng đến chín giờ tối",
        "Bãi giữ xe nằm ở phía sau khu giảng đường",
        "Căng tin ở tầng trệt của tòa nhà A",
      ],
      positive: [
        "Thư viện mới rộng rãi, có máy lạnh và nhiều chỗ ngồi yên tĩnh",
        "Phòng học sạch sẽ và máy chiếu rất rõ nét",
        "Wifi trong khuôn viên ổn định nên em làm bài trực tuyến rất thuận lợi",
        "Phòng máy được trang bị máy mới nên thực hành rất mượt",
        "Khu tự học có nhiều ổ cắm và đèn đủ sáng",
        "Căng tin sạch sẽ và phục vụ nhanh",
      ],
    },
    closers: {
      negative: [""],
      neutral: ["", ""],
      positive: ["", ", em rất hài lòng", " nên em thường học ở đây"],
    },
  },
  other: {
    subjects: [""],
    bodies: {
      negative: [
        "Thủ tục xin giấy xác nhận quá rườm rà, em phải đi lại nhiều lần",
        "Phòng đào tạo trả lời email rất chậm",
        "Lịch đăng ký học phần thay đổi liên tục mà không báo trước",
        "Học phí tăng nhưng nhà trường không giải thích rõ lý do",
        "Kết quả xét học bổng được công bố chưa minh bạch",
        "Cổng thông tin sinh viên hay bị lỗi vào giờ đăng ký",
        "Em gọi lên văn phòng khoa nhiều lần nhưng không ai nghe máy",
        "Đơn phúc khảo được xử lý quá lâu",
        "Lịch thi được thông báo quá sát ngày thi",
        "Việc đổi lịch học bù không được báo qua cổng thông tin",
      ],
      neutral: [
        "Em nộp học phí qua ngân hàng vào đầu học kỳ",
        "Lễ phát bằng tốt nghiệp được tổ chức vào cuối tháng sáu",
        "Văn phòng khoa làm việc từ thứ hai đến thứ sáu",
        "Thẻ sinh viên được cấp ở buổi sinh hoạt đầu khóa",
      ],
      positive: [
        "Phòng đào tạo hỗ trợ đăng ký môn học rất nhanh",
        "Các buổi tư vấn hướng nghiệp của trường rất hữu ích",
        "Cán bộ văn phòng khoa hướng dẫn thủ tục rõ ràng và lịch sự",
        "Hoạt động câu lạc bộ của trường rất phong phú",
        "Cổng thông tin sinh viên dễ dùng nên em tra cứu điểm rất nhanh",
      ],
    },
    closers: {
      negative: [""],
      neutral: ["", ""],
      positive: ["", ", em rất hài lòng", " nên em thấy yên tâm"],
    },
  },
};

const PREFIXES = ["", "", "", "Theo em, ", "Em thấy "] as const;

const FEEDBACK_GROUPS = ["Nhóm A", "Nhóm B", "Nhóm C", "Nhóm D"] as const;
type FeedbackGroup = (typeof FEEDBACK_GROUPS)[number];

/** Số câu tiêu cực / trung tính / tích cực dự định của từng chủ đề trong từng nhóm. */
const PLAN: Record<FeedbackGroup, Record<FeedbackTopic, readonly [number, number, number]>> = {
  "Nhóm A": {
    lecturer: [16, 2, 12],
    training_program: [22, 0, 1],
    facility: [8, 0, 3],
    other: [6, 0, 0],
  },
  "Nhóm B": {
    lecturer: [24, 1, 5],
    training_program: [22, 0, 1],
    facility: [9, 0, 2],
    other: [6, 0, 0],
  },
  "Nhóm C": {
    lecturer: [8, 3, 19],
    training_program: [22, 0, 1],
    facility: [8, 1, 2],
    other: [6, 0, 0],
  },
  "Nhóm D": {
    lecturer: [9, 3, 18],
    training_program: [22, 0, 1],
    facility: [7, 1, 3],
    other: [6, 0, 0],
  },
};

const TONES: readonly Tone[] = ["negative", "neutral", "positive"];
const MAX_ATTEMPTS = 1000;

/** Băm nhân (Knuth) để chọn vế "ngẫu nhiên" nhưng luôn cho cùng kết quả. */
function pick<T>(items: readonly T[], n: number, salt: number): T {
  const hashed = Math.imul(n + salt, 2654435761) >>> 0;
  return items[(hashed >>> 8) % items.length];
}

function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1);
}

function compose(topic: FeedbackTopic, tone: Tone, n: number): string {
  const { subjects, bodies, closers } = PHRASES[topic];
  const list = bodies[tone];
  const subject = subjects[n % subjects.length];
  const body = list[Math.floor(n / subjects.length) % list.length];
  const clause = subject === "" ? body : `${subject} ${body}`;
  const prefix = pick(PREFIXES, n, 7);
  const closer = pick(closers[tone], n, 13);
  return `${prefix}${prefix === "" ? clause : lowerFirst(clause)}${closer}.`;
}

/** Bộ sinh số giả ngẫu nhiên cố định hạt giống (mulberry32), chỉ để xáo thứ tự dòng cho tự nhiên. */
function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface FeedbackRow {
  text: string;
  group: string;
}

/** Các dòng của file, đã xáo thứ tự; `extras` (câu đặc biệt như không dấu, có số điện thoại) chèn rải đều. */
function feedbackRows(extras: readonly string[] = []): FeedbackRow[] {
  const seen = new Set<string>();
  const counters = new Map<string, number>();
  const rows: FeedbackRow[] = [];
  for (const group of FEEDBACK_GROUPS) {
    for (const topic of Object.keys(PHRASES) as FeedbackTopic[]) {
      TONES.forEach((tone, toneIndex) => {
        const key = `${topic}/${tone}`;
        for (let i = 0; i < PLAN[group][topic][toneIndex]; i += 1) {
          let n = counters.get(key) ?? 0;
          let text = compose(topic, tone, n);
          // Mỗi câu chỉ xuất hiện một lần trong file; trùng thì sang tổ hợp kế tiếp. Hết tổ hợp mà vẫn thiếu thì
          // báo lỗi rõ ràng (thêm vế câu hoặc giảm bảng số lượng) thay vì lặp mãi.
          let attempts = 0;
          while (seen.has(text)) {
            attempts += 1;
            if (attempts > MAX_ATTEMPTS) throw new Error(`Không đủ tổ hợp câu khác nhau cho nhóm ${key}`);
            n += 1;
            text = compose(topic, tone, n);
          }
          counters.set(key, n + 1);
          seen.add(text);
          rows.push({ text, group });
        }
      });
    }
  }

  const random = seededRandom(20261019);
  for (let i = rows.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }

  const step = Math.floor(rows.length / (extras.length + 1));
  extras.forEach((text, index) => {
    rows.splice((index + 1) * step + index, 0, { text, group: FEEDBACK_GROUPS[index % FEEDBACK_GROUPS.length] });
  });
  return rows;
}

const BOM = String.fromCharCode(0xfeff);

function csvCell(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}

const FEEDBACK_TEXT_COLUMN = "Góp ý";
// Tên cột chứa "minh hoạ" nên máy chủ gắn cờ `illustrative_groups`: nhóm A–D chỉ là cách chia cho ví dụ.
const FEEDBACK_GROUP_COLUMN = "Nhóm (minh hoạ)";

export interface FeedbackFile {
  name: string;
  mimeType: string;
  buffer: Buffer;
  /** Số dòng dữ liệu (không tính dòng tiêu đề). */
  rows: number;
}

/** CSV UTF-8 có BOM (như Excel xuất ra) với hai cột: câu góp ý và nhóm minh hoạ. */
export function feedbackFile(extras: readonly string[] = []): FeedbackFile {
  const rows = feedbackRows(extras);
  const lines = [
    `${csvCell(FEEDBACK_TEXT_COLUMN)},${csvCell(FEEDBACK_GROUP_COLUMN)}`,
    ...rows.map((row) => `${csvCell(row.text)},${csvCell(row.group)}`),
  ];
  return {
    name: "gop-y-tu-viet.csv",
    mimeType: "text/csv",
    buffer: Buffer.from(`${BOM}${lines.join("\n")}\n`, "utf-8"),
    rows: rows.length,
  };
}
