# Hosting miễn phí cho EduPulse: FE Next.js + BE FastAPI/ONNX (2026-10-15)

Nguồn: docs/trang giá chính thức của từng nhà cung cấp, đọc ngày 2026-10-15 (URL ở mục 8). Chưa deploy thử host nào; cold start chỉ là ước tính, chưa đo.

## 1. Kết luận
- **FE: Vercel Hobby** (docs không đòi thẻ, Next.js App Router native). Dự phòng: Netlify Free nếu cần SSR; nếu export tĩnh được thì Cloudflare Workers static assets / HF Static Space / GitHub Pages.
- **BE: không có host Docker miễn phí, đủ RAM (≥1,5-2 GB) mà không cần thẻ hoặc xác minh SV.** Ngoại lệ duy nhất là HF ZeroGPU (không thẻ) nhưng phải viết lại sang Gradio + GPU. Muốn link public ổn định phải có thẻ hoặc xác minh SV; không thì chỉ còn local + tunnel cho buổi demo.
- Đóng gói BE thành 1 Docker image trung lập (FastAPI + ORT int8 + JRE + model nướng sẵn; build `linux/amd64` vì M1 mặc định arm64) để đổi host không phải viết lại.

| Trường hợp | Chính | Dự phòng 1 | Dự phòng 2 |
|---|---|---|---|
| (a) Không thẻ, không SV | Local Mac + ngrok free (hoặc Tailscale Funnel), chỉ cho ngày demo | HF ZeroGPU Gradio (viết lại, không khuyến nghị) | Cloudflare Quick Tunnel (URL đổi mỗi lần chạy) |
| (b) Có thẻ | Google Cloud Run | Modal Starter | Oracle Always Free A1 (hoặc trả nhỏ: HF PRO $9, Fly.io ~$4-7/tháng) |
| (c) Xác minh được SV | Azure Container Apps trên Azure for Students (không thẻ) | Cloud Run nếu có thẻ | Local + ngrok |

FE chọn như nhau cho cả 3 trường hợp.

## 2. Ràng buộc lọc BE
- Cần Docker + JRE, RAM ≥1,5-2 GB (RSS 0,5-0,9 GB + dư), chỉ CPU, stateless nên hợp scale-to-zero nếu chịu được cold start.
- Loại ngay: Render free (512 MB, từ báo cáo r2), Railway Free (0,5 GB/service; trial chỉ 30 ngày/$5), PythonAnywhere free (disk 512 MiB, 100 CPU-giây/ngày, 1 web app hết hạn mỗi tháng, ASGI còn beta), Koyeb (cần thẻ; RAM free 10/2026 chưa xác minh, docs cũ ghi 512 MB).

## 3. So sánh FE
| Phương án | Chi phí / thẻ | Next.js App Router | Giới hạn chính, rủi ro |
|---|---|---|---|
| **Vercel Hobby** | $0, docs không nêu thẻ | Native, đủ SSR + route handler | Chỉ phi thương mại cá nhân (fair-use tính cả người được trả tiền viết code); 100 GB transfer, 1 triệu invocations, 4 CPU-giờ, 360 GB-giờ; vượt hạn mức thường phải chờ ~30 ngày; không có team collaboration (chỉ Pro) nên chỉ owner deploy |
| Netlify Free | $0 "forever", credit-based | Hỗ trợ Next.js (trang giá liệt kê); độ đầy đủ App Router chưa tự kiểm | Deploy 15 credit, bandwidth 20/GB, compute 10/GB-giờ, 2/10k request; credit Free/tháng chưa xác minh; hết credit thì pause (suy từ mục "Resume paused projects"); vừa đổi mô hình giá |
| Cloudflare Workers Free | $0 | vinext (CF khuyến nghị, **beta**) hoặc OpenNext (Next 16 + minor mới của 14/15) | 100k req/ngày, **CPU 10 ms/request**, RAM 128 MB, Worker ≤64 MiB; SSR dễ chạm 10 ms. Static assets miễn phí, không giới hạn nếu export tĩnh |
| HF Static Space | $0 cho mọi tài khoản ("never requires a paid plan") | Chỉ static export; có `app_build_command` + `app_file` để build Node | Không SSR; cùng hệ HF với repo model |
| GitHub Pages | $0 | Chỉ static export | Site ≤1 GB, 100 GB/tháng (soft), 10 builds/giờ (soft, không áp dụng với Actions riêng); cấm dùng cho kinh doanh; cần `basePath` nếu project site |

Static export không hỗ trợ tính năng cần server (đã thấy trong docs Next: dynamic route thiếu `generateStaticParams`, Route Handler dùng Request; còn lại theo docs: Server Actions, middleware, cookies, rewrites/redirects/headers, ISR). Cần chốt sớm: FE gọi BE từ trình duyệt (cần CORS) hay qua route handler.

## 4. So sánh BE
| Phương án | Chi phí / thẻ | RAM, CPU, Docker, Java | Sleep / cold start | Giới hạn chính, rủi ro |
|---|---|---|---|---|
| **Cloud Run** | Free/tháng (request-based): 180k vCPU-s + 360k GiB-s + 2M req ≈ 50 giờ active ở 1 vCPU/2 GiB; trial $300/90 ngày. **Cần billing account (thẻ)** | Chọn được (đề xuất 1 vCPU/2 GiB); Docker có, JRE trong image | Scale-to-zero; cold start = kéo image + JVM + ORT (ước 5-20 s, chưa đo) | Free tier tính như giảm giá theo giá Tier 1 nên region Tier 2 (vd. Singapore) được ít giây hơn (suy luận); không có hard cap chi tiêu: dùng `--max-instances=1` + budget alert |
| **Azure Container Apps** (Azure for Students) | $100 credit/12 tháng, **không thẻ**, đăng ký bằng school email, 18+ (theo trang Student Pack); ACA free 180k vCPU-s, 360k GiB-s, 2M req/subscription/tháng | Chọn được; Docker có | Scale-to-zero, cold start tương tự Cloud Run | Student sub có thể bị chính sách giới hạn region/dịch vụ (chưa kiểm); gia hạn hằng năm khi còn là SV; credit hết hạn sau 12 tháng |
| **Modal Starter** | $0 + compute, $30/tháng credit; **bắt buộc có payment method on file** | Chọn được; image khai báo bằng Python SDK | Scale-to-zero (cold start thường ngắn hơn, chưa kiểm) | Lock-in SDK `modal`; vượt credit tự tính tiền (đặt spend budget); có credit học thuật tới $10k cho grad student/lab/researcher (chưa kiểm điều kiện) |
| **Oracle Always Free A1** | $0; **cần thẻ credit/debit thật** (không nhận prepaid, virtual, debit có PIN) | Docs hiện ghi 2 OCPU Arm + 12 GB tổng (trước đây tôi biết là 4/24); Docker arm64, JRE qua apt (wheel jdk4py arm64 chưa kiểm) | Luôn bật, không cold start | Bị coi idle và thu hồi nếu 7 ngày liền CPU p95 <20% **và** network <20% **và** memory <20% (A1): demo ít tải dễ dính, cấp RAM nhỏ (2-3 GB) để memory >20% (suy luận, chưa thử); "out of host capacity"; không SLA; tự lo TLS |
| Fly.io | Trial 2 giờ hoặc 7 ngày, sau đó cần thẻ; 512 MB $3,69/tháng, 1 GB $6,70/tháng (chạy liên tục) | shared-cpu-1x; Docker có | auto-stop thì chỉ trả khi chạy | Không free |
| Railway | Free: trial 30 ngày/$5 rồi $1/tháng, không thẻ | 1 vCPU, 0,5 GB/service | - | Loại: 0,5 GB < RSS; RAM $10/GB/tháng |
| Koyeb | Banner "joining Mistral AI"; FAQ hỏi thẻ/account validation; ví dụ billing có "1 free Instance" | Spec free 10/2026 chưa xác minh | Scale-to-zero có trong docs | Loại tạm; tương lai free tier chưa rõ sau khi Mistral mua |
| HF ZeroGPU (Gradio) | $0, không thẻ; ≤2 Space cho tài khoản free (email xác minh, >30 ngày) | Chỉ Gradio SDK; cần hàm `@spaces.GPU`; apt qua `packages.txt` (chưa kiểm trên ZeroGPU) | Không có số cold start/sleep công bố | Quota GPU tính theo người gọi: ẩn danh 2 / free 5 / PRO 40 phút mỗi ngày; gọi qua server bằng token chủ thì mọi khách dùng chung 5 phút (suy luận); phải viết lại sang Gradio + torch GPU (2-4 ngày theo r2); dùng GPU chỉ để thỏa điều kiện host nên rủi ro chính sách |
| Local + tunnel | $0, không thẻ | Mac 16 GB dư; chạy trực tiếp | Không sleep nhưng máy phải bật + mạng ổn định | Quick Tunnel: không cần account/domain, URL tạm, "testing and development", ≤200 in-flight (429), không SSE. ngrok free: 1 dev domain cố định, 1 GB + 20k req/tháng, interstitial cho HTML. Tailscale Funnel: URL `*.ts.net` cố định, beta, băng thông giới hạn không chỉnh được. Cloudflare Tunnel đặt tên: cần domain trong Cloudflare + onboarding Zero Trust đòi thông tin thanh toán dù gói Free nên không hợp (a) |
| GitHub Student Pack | Không phải host: Azure $100 (không thẻ, 18+), Heroku $13/tháng × 24 tháng, Namecheap .me 1 năm | Heroku chưa đánh giá RAM/Docker | - | Hữu ích nhất là Azure; domain .me để gắn custom domain |

## 5. Xếp hạng và lý do
**(a) Không thẻ, không SV**
1. Local + ngrok/Tailscale: $0, đủ RAM, URL cố định; đổi lại chỉ sống khi Mac bật và có cap 1 GB / 20k req/tháng (ngrok). Hợp buổi bảo vệ, không phải hosting thật. Env của Vercel trỏ vào URL tunnel.
2. HF ZeroGPU: chỉ khi bắt buộc cần link sống 24/7 không thẻ và chấp nhận viết lại; r2 đã không khuyến nghị.
3. Quick Tunnel: chữa cháy.
Nhận xét: mọi host đủ RAM đều đứng sau thẻ hoặc xác minh SV; nếu cần link sống thật, mở khóa 1 trong 2 là quyết định đáng làm nhất.

**(b) Có thẻ**
1. Cloud Run: free tier thường trực, Docker chuẩn, scale-to-zero nên chi phí kỳ vọng $0 với traffic demo. Rủi ro: không hard cap chi tiêu (giảm bằng max-instances=1 + budget alert), cold start.
2. Modal: dựng nhanh, có spend budget; đổi lại lock-in SDK + thẻ bắt buộc.
3. Oracle A1: luôn bật, nhiều RAM; đổi lại tốn công vận hành (TLS, patch), có thể bị thu hồi khi idle, thiếu capacity, thẻ VN có thể bị từ chối. Cần không cold start mà ngại vận hành: trả nhỏ HF PRO $9 (cpu-basic 16 GB, theo r2) hoặc Fly.io ~$4-7/tháng.

**(c) Xác minh được SV**
1. Azure Container Apps (Azure for Students): $100 + free grants, không thẻ nên không có rủi ro hóa đơn, scale-to-zero. Đổi lại phụ thuộc xác minh school email (trượt thì quay về (b) hoặc (a)), chính sách region của student sub chưa kiểm.
2. Cloud Run nếu có thẻ; không thì local + ngrok.

**FE (mọi trường hợp):** Vercel Hobby, rồi Netlify Free, rồi export tĩnh lên Cloudflare Workers assets / HF Static.

## 6. Lưu ý triển khai và rủi ro
- CORS: `CORSMiddleware` cho origin của FE; hoặc proxy qua route handler (timeout function Hobby chưa xác minh).
- Cold start: FE hiển thị "server đang khởi động" + retry/backoff; ping `/health` ~2 phút trước demo; cron ping giữ ấm gần như miễn phí ở request-based billing nhưng không được đảm bảo.
- Nướng model 284 MB vào image thay vì tải từ Hub lúc start: cold start nhanh hơn, đổi lại image ~0,8-1 GB.
- Mọi nền tảng tính theo mức dùng cần chặn trần chi (Cloud Run max-instances, Modal spend budget).
- Chính sách đổi nhanh (HF 07/2026, Oracle giảm A1, Koyeb bị mua, Netlify sang credit-based): xác nhận lại ngay trước khi deploy.

## 7. Giới hạn nghiên cứu
- Không deploy thử, không đo RSS/cold start trên host thật. Mỗi host chủ yếu dựa 1-2 trang docs chính thức (nguồn mạnh nhất cho giá/hạn mức), chưa có nguồn độc lập thứ ba (forum, case study).
- Chưa khảo sát: AWS (Lambda container), DigitalOcean, Azure App Service, Heroku (spec dyno), region policy của Azure student, timeout function Vercel Hobby, credit Free của Netlify.
- Trang đọc bằng curl + tách text nên có thể sót nội dung JS/accordion (vd. câu trả lời FAQ Koyeb); docs eligibility của GitHub Education trả 404 nên chưa đọc.

## 8. Nguồn (đọc 2026-10-15)
- Vercel: https://vercel.com/docs/plans/hobby ; https://vercel.com/docs/limits/fair-use-guidelines
- Netlify: https://www.netlify.com/pricing/
- Cloudflare Workers: https://developers.cloudflare.com/workers/platform/limits/index.md ; https://developers.cloudflare.com/workers/framework-guides/web-apps/nextjs/index.md ; https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/index.md ; https://opennext.js.org/cloudflare
- Cloudflare Tunnel: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/index.md ; https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/index.md ; https://developers.cloudflare.com/cloudflare-one/setup/index.md
- HF: https://huggingface.co/docs/hub/spaces-sdks-static ; https://huggingface.co/docs/hub/spaces-zerogpu
- GitHub Pages: https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits ; Next.js static export: https://nextjs.org/docs/app/guides/static-exports
- Cloud Run: https://cloud.google.com/run/pricing ; https://cloud.google.com/free/docs/free-cloud-features
- Azure: https://azure.microsoft.com/en-us/free/students/ ; https://learn.microsoft.com/en-us/azure/container-apps/billing ; GitHub Student Pack: https://education.github.com/pack
- Oracle: https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm ; https://www.oracle.com/cloud/free/faq/
- Modal: https://modal.com/pricing ; https://modal.com/docs/guide/billing
- Fly.io: https://fly.io/docs/about/pricing/ ; https://fly.io/docs/about/free-trial/
- Railway: https://docs.railway.com/reference/pricing/plans ; https://railway.com/pricing
- Koyeb: https://www.koyeb.com/docs/faqs/pricing ; https://www.koyeb.com/pricing
- PythonAnywhere: https://help.pythonanywhere.com/pages/FreeAccountsFeatures/ ; https://help.pythonanywhere.com/pages/ASGICommandLine/
- ngrok: https://ngrok.com/docs/pricing-limits/free-plan-limits/ ; Tailscale Funnel: https://tailscale.com/kb/1223/funnel
- Báo cáo trước (Render 512 MB, quota ZeroGPU, HF PRO $9): plans/reports/researcher-261014-1913-hf-spaces-streamlit-deploy.md

## Câu hỏi chưa giải quyết
1. Cần link sống 24/7 hay chỉ trong buổi demo/bảo vệ? (quyết định (a) local là đủ hay phải mở khóa bằng thẻ/SV)
2. User có thẻ không (credit hay debit VN; Oracle không nhận prepaid/virtual) và có đủ điều kiện Azure for Students / GitHub Education (school email, 18+) không?
3. FE có cần SSR/route handler hay export tĩnh được? (quyết định Vercel/Netlify hay Cloudflare/HF Static)
4. Có ai trong nhóm được trả tiền cho dự án này, và có nhiều người cùng deploy không? (điều khoản non-commercial và việc Hobby không có team collaboration)
5. Koyeb free instance 10/2026: RAM bao nhiêu, còn nhận tài khoản mới không?
6. Azure student subscription cho tạo Container Apps ở region nào; provider Microsoft.App đăng ký được không?
7. Cold start thật của image (JRE + ORT 284 MB) trên Cloud Run/ACA: đo sau khi có Dockerfile.
8. Netlify Free: bao nhiêu credit/tháng; timeout function của Vercel Hobby.
