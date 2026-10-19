// Câu thay cho golden g01 khi ghi fixture của web. g01 có một cụm 9 từ liên tiếp trùng với một câu của UIT-VSFC
// nên không đưa vào repo qua fixture; câu này tự viết, cùng nhãn (tích cực, giảng viên) và không bị gắn cờ.
export const G01_REPLACEMENT = "Thầy giảng bài dễ hiểu, em thấy thầy rất tận tâm với lớp.";

/** Các câu golden dùng cho fixture, đã thay những câu không được đưa vào repo. */
export function fixtureSentences(items) {
  return items.map((item) => (item.id === "g01" ? { ...item, text: G01_REPLACEMENT } : item));
}
