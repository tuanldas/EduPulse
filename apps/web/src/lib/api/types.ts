// Tên gọn cho kiểu sinh từ OpenAPI (`schema.d.ts`, do `make gen-api` ghi). Chỉ đặt bí danh,
// không khai báo lại hình dạng dữ liệu: đổi hợp đồng thì TypeScript báo lỗi ở chỗ dùng.
import type { components } from "./schema";

type Schemas = components["schemas"];

export type AppConfig = Schemas["AppConfig"];
export type AnalyzeResponse = Schemas["AnalyzeResponse"];
export type BaselineComparison = Schemas["BaselineComparison"];
export type ExplainResponse = Schemas["ExplainResponse"];
export type HeadExplanation = Schemas["HeadExplanation"];
export type TokenEffect = Schemas["TokenEffect"];
export type FlagReason = Schemas["FlagReason"];
export type ModelCard = Schemas["ModelCard"];
export type Health = Schemas["Health"];

export type SentimentLabel = Schemas["SentimentPrediction"]["label"];
export type TopicLabel = Schemas["TopicPrediction"]["label"];
export type PiiType = Schemas["AnalyzeResponse"]["pii_types"][number];
export type Stage = Schemas["ModelInfo"]["stage"];
