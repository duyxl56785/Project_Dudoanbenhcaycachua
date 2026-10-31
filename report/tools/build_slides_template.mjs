import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const outDir = path.join(root, ".report_build");
const previewDir = path.join(root, ".pptx_render");
const assetsDir = path.join(root, "report", "assets");
await fs.mkdir(outDir, { recursive: true });
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

async function addJpeg(slide, relativePath, alt, position) {
  const imagePath = path.join(assetsDir, relativePath);
  const bytes = await fs.readFile(imagePath);
  slide.images.add({
    blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    contentType: "image/jpeg",
    alt,
    fit: "contain",
    position,
    geometry: "rect",
  });
}

function slideBase(title, kicker = "NHẬN DIỆN BỆNH LÁ CÀ CHUA") {
  const slide = deck.slides.add();
  slide.background.fill = colors.white;
  box(slide, kicker, 68, 40, 600, 24, { size: 13, color: colors.blue, bold: true });
  box(slide, title, 68, 78, 1120, 66, { size: 34, color: colors.navy, bold: true });
  box(slide, "Nhóm 4 · Học máy và Khai phá dữ liệu · NV5 tổng hợp", 68, 676, 900, 20, { size: 11, color: colors.muted });
  return slide;
}

let slide = deck.slides.add();
slide.background.fill = colors.pale;
box(slide, "NHẬN DIỆN BỆNH LÁ CÀ CHUA", 84, 118, 840, 30, { size: 17, color: colors.blue, bold: true });
box(slide, "So sánh CNN và ResNet18", 84, 170, 980, 90, { size: 46, color: colors.navy, bold: true });
box(slide, "Báo cáo bài tập lớn · Nhóm 4 · Trường Đại học CMC", 88, 288, 760, 34, { size: 22, color: colors.muted });
box(slide, "CNN và ResNet18 đã kiểm chứng trên Test chung", 88, 550, 760, 30, { size: 16, color: colors.ink });

slide = slideBase("Mục tiêu và phạm vi");
box(slide, "Mục tiêu", 86, 175, 420, 34, { size: 25, color: colors.blue, bold: true });
box(slide, "Phân loại ảnh lá cà chua vào bốn lớp đã thống nhất.\nSo sánh CNN huấn luyện từ đầu với ResNet18 dùng Transfer Learning.\nĐối chiếu kết quả với demo Streamlit.", 86, 225, 500, 190, { size: 22 });
box(slide, "Phạm vi", 700, 175, 420, 34, { size: 25, color: colors.blue, bold: true });
box(slide, "Kết quả chỉ áp dụng cho các lớp và dữ liệu đã học.\nPlantVillage có điều kiện chụp được kiểm soát.\nChưa có thí nghiệm riêng cho bệnh mới chớm.", 700, 225, 470, 190, { size: 22 });

slide = slideBase("Dữ liệu và pipeline");
box(slide, "Nguồn dữ liệu", 86, 175, 360, 30, { size: 24, color: colors.blue, bold: true });
box(slide, "PlantVillage · 6.502 ảnh\nTrain 4.115 · Validation 1.031 · Test 1.356", 86, 220, 500, 80, { size: 20 });
box(slide, "Pipeline chung", 86, 350, 360, 30, { size: 24, color: colors.blue, bold: true });
box(slide, "Manifest → transform → model → prediction CSV → đánh giá NV5", 86, 395, 900, 42, { size: 24, color: colors.ink, bold: true });
box(slide, "Mapping 0–3: Healthy · Early Blight · Late Blight · Bacterial Spot\nNguồn: PR #8 và configs/class_mapping.json", 86, 500, 1000, 52, { size: 18, color: colors.muted });

slide = slideBase("Hai hướng mô hình");
box(slide, "CNN tự xây dựng", 86, 182, 500, 34, { size: 26, color: colors.blue, bold: true });
box(slide, "Học từ đầu · 4 khối Conv–BN–ReLU–MaxPool\nGAP · Dropout 0,3 · Linear 4 lớp\nCheckpoint E1 epoch 24 chọn theo Validation", 86, 238, 520, 160, { size: 21 });
box(slide, "ResNet18 Transfer Learning", 700, 182, 500, 34, { size: 26, color: colors.blue, bold: true });
box(slide, "ImageNet pretrained · fine-tuning layer4\nCheckpoint epoch 9 chọn theo Validation\nTest artifact và SHA-256 đã kiểm", 700, 238, 500, 160, { size: 21 });

slide = slideBase("Giao thức đánh giá");
box(slide, "NV5 dùng scripts/evaluate_models.py để kiểm tra hai prediction CSV có cùng sample ID và ground truth.", 86, 180, 1060, 50, { size: 23, bold: true });
box(slide, "Chỉ đánh giá Test sau khi checkpoint đã được chọn bằng Validation.", 86, 270, 980, 40, { size: 22, color: colors.blue, bold: true });
box(slide, "Metrics cần báo cáo", 86, 385, 360, 30, { size: 24, color: colors.blue, bold: true });
box(slide, "Accuracy\nMacro Precision / Recall / F1\nF1 từng lớp\nSố tham số và thời gian đo", 86, 430, 430, 150, { size: 21 });
box(slide, "Nguồn số liệu", 700, 385, 360, 30, { size: 24, color: colors.blue, bold: true });
box(slide, "results/comparison.csv\nresults/analysis.md\nHai confusion matrix\nPrediction CSV và metadata checkpoint", 700, 430, 430, 150, { size: 21 });

slide = slideBase("Kết quả so sánh");
box(slide, "Test chung 1.356 ảnh · Artifact CNN PR #9 đã đối chiếu dưới 1e-6", 86, 190, 1000, 44, { size: 22, color: colors.muted, bold: true });
const resultTable = slide.tables.add({
  rows: 3,
  columns: 5,
  left: 86,
  top: 245,
  width: 1090,
  height: 135,
  values: [
    ["Mô hình", "Accuracy", "Macro F1", "Đúng/Test", "Tham số"],
    ["CNN", "96,9764%", "96,5034%", "1.315/1.356", "389.924"],
    ["ResNet18", "98,6726%", "98,3978%", "1.338/1.356", "11.178.564"],
  ],
});
resultTable.styleOptions = { headerRow: true, bandedRows: true };
resultTable.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 5 }).assign({
  fill: colors.navy,
  textStyle: { color: colors.white, bold: true, fontSize: 17, fontFamily: "Aptos" },
});
resultTable.cells.block({ row: 1, column: 0, rowCount: 2, columnCount: 5 }).assign({
  textStyle: { color: colors.ink, fontSize: 16, fontFamily: "Aptos" },
  fill: colors.pale,
});
resultTable.borders.assign({ style: "solid", fill: "#D9D9D9", width: 1 });
slide.charts.add("bar", {
  position: { left: 260, top: 415, width: 760, height: 205 },
  title: "Macro F1 trên Test",
  titleTextStyle: { fontSize: 18, bold: true, fill: colors.ink },
  categories: ["CNN", "ResNet18"],
  // Biểu đồ hiển thị làm tròn 0,1 điểm phần trăm; bảng phía trên giữ số đã kiểm chứng đầy đủ.
  series: [{ name: "Macro F1", values: [0.965, 0.984], fill: colors.blue }],
  barOptions: { direction: "column", grouping: "clustered", gapWidth: 70 },
  hasLegend: false,
  xAxis: { textStyle: { fontSize: 15, fill: colors.ink }, line: { style: "solid", fill: "#D9D9D9", width: 1 } },
  yAxis: { min: 0.94, max: 1.0, majorUnit: 0.02, numberFormatCode: "0%", textStyle: { fontSize: 12, fill: colors.muted }, majorGridlines: { style: "solid", fill: "#D9D9D9", width: 1 } },
  dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.00%", textStyle: { fontSize: 13, fill: colors.ink, bold: true } },
});

slide = slideBase("Phân tích nhầm lẫn");
box(slide, "CNN confusion matrix\nH: 329  0  1  0\nE: 0  198  14  3\nL: 5  12  358  0\nB: 0  2  4  430", 86, 190, 470, 200, { size: 20, color: colors.ink, bold: true });
box(slide, "ResNet18 confusion matrix\nH: 329  0  1  0\nE: 3  204  6  2\nL: 0  3  372  0\nB: 0  2  1  433", 680, 190, 470, 200, { size: 20, color: colors.ink, bold: true });
box(slide, "Lỗi: CNN 41 ảnh · ResNet18 18 ảnh; Early Blight → Late Blight còn khó nhất", 86, 470, 1050, 40, { size: 21, color: colors.blue, bold: true });
box(slide, "H = Healthy · E = Early Blight · L = Late Blight · B = Bacterial Spot", 86, 530, 1050, 35, { size: 17, color: colors.muted });

slide = slideBase("Demo Streamlit với hai checkpoint thật");
await addJpeg(
  slide,
  path.join("nv4_demo", "cnn_resnet18_compare_20261030.jpg"),
  "Ảnh chụp Streamlit NV4 so sánh CNN và ResNet18 trên cùng một ảnh Bacterial Spot",
  { left: 86, top: 165, width: 690, height: 390 },
);
box(slide, "Kết quả trên một ảnh Test", 825, 180, 330, 32, { size: 23, color: colors.blue, bold: true });
box(slide, "CNN: Bacterial Spot 96,95%\nResNet18: Bacterial Spot 67,11%\nCả hai strict-load CPU\n--model both: max |delta| = 0", 825, 230, 330, 180, { size: 19 });
box(slide, "Nguồn: ảnh chụp NV4, commit 2406b12. Đây là minh họa giao diện, không phải Accuracy toàn tập. Video chưa bàn giao.", 86, 585, 1070, 45, { size: 15, color: colors.muted });

slide = slideBase("Kết luận và hướng phát triển");
box(slide, "Kết luận", 86, 185, 280, 32, { size: 25, color: colors.blue, bold: true });
box(slide, "ResNet18 đạt macro-F1 0,983978, cao hơn CNN 0,965034 trên 1.356 ảnh Test.\nKết quả chỉ áp dụng PlantVillage.", 86, 235, 560, 120, { size: 21 });
box(slide, "Hướng phát triển", 700, 185, 320, 32, { size: 25, color: colors.blue, bold: true });
box(slide, "Bổ sung ảnh thực địa.\nKiểm tra độ tin cậy và calibration.\nĐánh giá bệnh mới chớm khi có dataset phù hợp.", 700, 235, 450, 160, { size: 22 });
box(slide, "Câu hỏi phản biện: xem docs/NV5_DEFENSE_QA.md", 86, 550, 800, 30, { size: 18, color: colors.muted });

slide = slideBase("F1 từng lớp");
box(slide, "CNN\nHealthy: 0,990964\nEarly Blight: 0,927400\nLate Blight: 0,952128\nBacterial Spot: 0,989643", 100, 185, 440, 205, { size: 20, color: colors.ink, bold: true });
box(slide, "ResNet18\nHealthy: 0,993958\nEarly Blight: 0,962264\nLate Blight: 0,985430\nBacterial Spot: 0,994259", 650, 185, 470, 205, { size: 20, color: colors.ink, bold: true });
box(slide, "ResNet18 cải thiện F1 ở cả bốn lớp. Cần đọc cùng support và confusion matrix, không chỉ dựa Accuracy.", 100, 450, 1000, 72, { size: 22, color: colors.blue });
box(slide, "Nguồn: results/cnn_metrics_nv2_handoff.json và results/resnet18_metrics_nv3_handoff.json", 100, 570, 980, 24, { size: 15, color: colors.muted });

slide = slideBase("Nguồn số liệu và giới hạn");
box(slide, "Nguồn artifact", 86, 180, 340, 34, { size: 25, color: colors.blue, bold: true });
box(slide, "PR #9 · cnn_metrics.json · prediction CSV\nPR #7 ZIP · SHA-256 checkpoint ResNet18\nNV5 đối chiếu metrics dưới sai số 1e-6", 86, 235, 520, 130, { size: 19 });
box(slide, "Giới hạn", 700, 180, 340, 34, { size: 25, color: colors.blue, bold: true });
box(slide, "PlantVillage có điều kiện kiểm soát\nẢnh ngoài bốn lớp vẫn bị ép nhãn\nChưa có Test bệnh mới chớm", 700, 235, 470, 130, { size: 21 });
box(slide, "Không dùng Test để chọn checkpoint hoặc hyperparameter.", 86, 500, 1000, 36, { size: 22, color: colors.navy, bold: true });

slide = slideBase("Bàn giao cần thiết để chốt báo cáo");
box(slide, "NV3", 86, 180, 260, 34, { size: 25, color: colors.blue, bold: true });
box(slide, "Đã nhận: checkpoint/hash, CSV, metrics\nCần duy trì link ZIP và provenance\nKhông commit checkpoint nhị phân vào NV5", 86, 235, 500, 150, { size: 20 });
box(slide, "NV4 và nhóm trưởng", 700, 180, 390, 34, { size: 25, color: colors.blue, bold: true });
box(slide, "Ảnh/video và checkpoint demo\nMẫu Word/PPTX, người trình bày\nReview số liệu trước khi merge", 700, 235, 440, 150, { size: 20 });
box(slide, "Đã chạy evaluate_models.py với hai CSV. Mọi cập nhật sau dùng comparison.csv từ cùng nguồn.", 86, 510, 1080, 58, { size: 21, color: colors.navy, bold: true });

for (const [index, item] of deck.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await deck.export({ slide: item, format: "png", scale: 1 });
  await fs.writeFile(path.join(previewDir, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
}
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(path.join(outDir, "slides_nhom_tomato_two_models_nv5.pptx"));
console.log(path.join(outDir, "slides_nhom_tomato_two_models_nv5.pptx"));
