import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const outDir = path.join(root, "report", "deliverables");
const previewDir = path.join(root, ".pptx_render");
await fs.mkdir(previewDir, { recursive: true });

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const colors = { navy: "#1F4E79", blue: "#2F5597", ink: "#17202A", muted: "#5B6573", pale: "#F3F6FA", white: "#FFFFFF" };

function box(slide, text, x, y, w, h, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { fontSize: style.size ?? 22, color: style.color ?? colors.ink, bold: style.bold ?? false, fontFamily: "Aptos" };
  return shape;
}

function slideBase(title, kicker = "NHẬN DIỆN BỆNH LÁ CÀ CHUA") {
  const slide = deck.slides.add();
  slide.background.fill = colors.white;
  box(slide, kicker, 68, 40, 600, 24, { size: 13, color: colors.blue, bold: true });
  box(slide, title, 68, 78, 1120, 66, { size: 34, color: colors.navy, bold: true });
  box(slide, "Nhóm 4 · Học máy và Khai phá dữ liệu · Bản khung NV5", 68, 676, 900, 20, { size: 11, color: colors.muted });
  return slide;
}

let slide = deck.slides.add();
slide.background.fill = colors.pale;
box(slide, "NHẬN DIỆN BỆNH LÁ CÀ CHUA", 84, 118, 840, 30, { size: 17, color: colors.blue, bold: true });
box(slide, "So sánh CNN và ResNet18", 84, 170, 980, 90, { size: 46, color: colors.navy, bold: true });
box(slide, "Báo cáo bài tập lớn · Nhóm 4 · Trường Đại học CMC", 88, 288, 760, 34, { size: 22, color: colors.muted });
box(slide, "Bản khung chờ số liệu Test và minh chứng từ NV2, NV3, NV4", 88, 550, 760, 30, { size: 16, color: colors.ink });

slide = slideBase("Mục tiêu và phạm vi");
box(slide, "Mục tiêu", 86, 175, 420, 34, { size: 25, color: colors.blue, bold: true });
box(slide, "Phân loại ảnh lá cà chua vào bốn lớp đã thống nhất.\nSo sánh CNN huấn luyện từ đầu với ResNet18 dùng Transfer Learning.\nĐối chiếu kết quả với demo Streamlit.", 86, 225, 500, 190, { size: 22 });
box(slide, "Phạm vi", 700, 175, 420, 34, { size: 25, color: colors.blue, bold: true });
box(slide, "Kết quả chỉ áp dụng cho các lớp và dữ liệu đã học.\nPlantVillage có điều kiện chụp được kiểm soát.\nChưa có thí nghiệm riêng cho bệnh mới chớm.", 700, 225, 470, 190, { size: 22 });

slide = slideBase("Dữ liệu và pipeline");
box(slide, "Nguồn dữ liệu", 86, 175, 360, 30, { size: 24, color: colors.blue, bold: true });
box(slide, "[Điền nguồn, số ảnh từng lớp và phiên bản manifest từ NV1]", 86, 220, 440, 80, { size: 20 });
box(slide, "Pipeline chung", 86, 350, 360, 30, { size: 24, color: colors.blue, bold: true });
box(slide, "Manifest → transform → model → prediction CSV → đánh giá NV5", 86, 395, 900, 42, { size: 24, color: colors.ink, bold: true });
box(slide, "Mapping nhãn duy nhất: Healthy · Early Blight · Late Blight · Bacterial Spot", 86, 500, 1000, 32, { size: 18, color: colors.muted });

slide = slideBase("Hai hướng mô hình");
box(slide, "CNN tự xây dựng", 86, 182, 500, 34, { size: 26, color: colors.blue, bold: true });
box(slide, "• Học từ đầu\n• [Điền kiến trúc, optimizer và seed]\n• Chọn checkpoint bằng Validation", 86, 238, 520, 160, { size: 22 });
box(slide, "ResNet18 Transfer Learning", 700, 182, 500, 34, { size: 26, color: colors.blue, bold: true });
box(slide, "• Backbone pretrained ImageNet\n• Feature extraction và fine-tuning\n• Chọn stage bằng Validation", 700, 238, 500, 160, { size: 22 });

slide = slideBase("Giao thức đánh giá");
box(slide, "NV5 dùng scripts/evaluate_models.py để kiểm tra hai prediction CSV có cùng sample ID và ground truth.", 86, 180, 1060, 50, { size: 23, bold: true });
box(slide, "Chỉ đánh giá Test sau khi checkpoint đã được chọn bằng Validation.", 86, 270, 980, 40, { size: 22, color: colors.blue, bold: true });
box(slide, "Metrics cần báo cáo", 86, 385, 360, 30, { size: 24, color: colors.blue, bold: true });
box(slide, "Accuracy\nMacro Precision / Recall / F1\nF1 từng lớp\nSố tham số và thời gian đo", 86, 430, 430, 150, { size: 21 });
box(slide, "Nguồn số liệu", 700, 385, 360, 30, { size: 24, color: colors.blue, bold: true });
box(slide, "results/comparison.csv\nresults/analysis.md\nHai confusion matrix\nPrediction CSV và metadata checkpoint", 700, 430, 430, 150, { size: 21 });

slide = slideBase("Kết quả so sánh");
box(slide, "[Chèn bảng results/comparison.csv sau khi có prediction CSV thật]", 86, 190, 1000, 44, { size: 24, color: colors.muted, bold: true });
box(slide, "CNN", 100, 300, 250, 38, { size: 24, color: colors.blue, bold: true });
box(slide, "Accuracy: [điền]\nMacro F1: [điền]\nTham số: [điền]", 100, 350, 350, 130, { size: 22 });
box(slide, "ResNet18", 650, 300, 300, 38, { size: 24, color: colors.blue, bold: true });
box(slide, "Accuracy: [điền]\nMacro F1: [điền]\nTham số: [điền]", 650, 350, 350, 130, { size: 22 });

slide = slideBase("Phân tích nhầm lẫn");
box(slide, "[Chèn confusion matrix CNN]", 86, 190, 470, 180, { size: 25, color: colors.muted, bold: true });
box(slide, "[Chèn confusion matrix ResNet18]", 680, 190, 470, 180, { size: 25, color: colors.muted, bold: true });
box(slide, "Cặp lớp dễ nhầm: [điền từ results/analysis.md]", 86, 470, 1000, 40, { size: 23, color: colors.blue, bold: true });
box(slide, "Ví dụ ảnh Test đúng/sai: [điền path và giải thích ngắn]", 86, 530, 1000, 35, { size: 20 });

slide = slideBase("Demo và giới hạn");
box(slide, "Demo", 86, 185, 280, 32, { size: 25, color: colors.blue, bold: true });
box(slide, "[Chèn ảnh màn hình Streamlit và lệnh chạy từ NV4]", 86, 235, 500, 120, { size: 22 });
box(slide, "Giới hạn", 700, 185, 280, 32, { size: 25, color: colors.blue, bold: true });
box(slide, "Không thay thế chuyên gia.\nẢnh ngoài miền có thể bị ép vào một lớp.\nChưa chứng minh phát hiện sớm.", 700, 235, 420, 150, { size: 22 });

slide = slideBase("Kết luận và hướng phát triển");
box(slide, "Kết luận", 86, 185, 280, 32, { size: 25, color: colors.blue, bold: true });
box(slide, "[Tóm tắt mô hình phù hợp hơn dựa trên số liệu thật và điều kiện đo]", 86, 235, 530, 100, { size: 22 });
box(slide, "Hướng phát triển", 700, 185, 320, 32, { size: 25, color: colors.blue, bold: true });
box(slide, "Bổ sung ảnh thực địa.\nKiểm tra độ tin cậy và calibration.\nĐánh giá bệnh mới chớm khi có dataset phù hợp.", 700, 235, 450, 160, { size: 22 });
box(slide, "Câu hỏi phản biện: xem docs/NV5_DEFENSE_QA.md", 86, 550, 800, 30, { size: 18, color: colors.muted });

for (const [index, item] of deck.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await deck.export({ slide: item, format: "png", scale: 1 });
  await fs.writeFile(path.join(previewDir, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
}
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(path.join(outDir, "slides_nhom_tomato_template.pptx"));
console.log(path.join(outDir, "slides_nhom_tomato_template.pptx"));
