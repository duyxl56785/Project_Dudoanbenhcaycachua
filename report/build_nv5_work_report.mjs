import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "report", "bao_cao_cong_viec_nv5.pptx");
const preview = path.join(root, ".nv5_slide_preview.png");

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slide = deck.slides.add();
const blue = "#2F66B3";
const headerBlue = "#1388C5";
const lightBlue = "#DCE6F2";
const lighter = "#F0F4FA";
const dark = "#1F4E79";
const red = "#B83A2E";
const gray = "#4A5568";

slide.background.fill = "#FFFFFF";

function shape(geometry, x, y, w, h, fill, lineFill = fill) {
  return slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: lineFill, width: 1 },
  });
}

function text(value, x, y, w, h, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = value;
  box.text.style = {
    fontFamily: "Aptos",
    fontSize: options.size ?? 17,
    color: options.color ?? dark,
    bold: options.bold ?? false,
    align: options.align ?? "left",
    verticalAlign: options.verticalAlign ?? "middle",
  };
  return box;
}

// Header and title area inspired by the supplied CMC work-report slide.
shape("rect", 0, 0, 1280, 58, headerBlue, headerBlue);
text("BÁO CÁO CÔNG VIỆC: NHẬN DIỆN BỆNH QUA ẢNH LÁ CÀ CHUA", 44, 8, 1050, 40, { size: 25, color: "#FFFFFF", bold: true, align: "center" });
text("Sinh viên: Phạm Thế Cường", 62, 77, 560, 32, { size: 22, color: red, bold: true });
text("Nhiệm vụ NV5 – Đánh giá, báo cáo và slide", 62, 112, 750, 26, { size: 16, color: gray });
text("Nhóm 4 · Học máy và Khai phá dữ liệu", 920, 93, 290, 26, { size: 15, color: blue, bold: true, align: "right" });

const x0 = 62;
const y0 = 160;
const widths = [58, 252, 540, 130, 190];
const headerH = 58;
const rowH = 58;
const totalW = widths.reduce((a, b) => a + b, 0);
const headers = ["STT", "NỘI DUNG CÔNG VIỆC", "KẾT QUẢ", "TỰ ĐÁNH GIÁ", "TRƯỞNG NHÓM\nĐÁNH GIÁ"];
let x = x0;
for (let i = 0; i < widths.length; i++) {
  shape("rect", x, y0, widths[i], headerH, blue, "#FFFFFF");
  text(headers[i], x + 5, y0 + 6, widths[i] - 10, headerH - 12, { size: 15, color: "#FFFFFF", bold: true, align: "center" });
  x += widths[i];
}

const rows = [
  ["1", "Thiết kế script đánh giá thống nhất", "Hoàn thiện `scripts/evaluate_models.py`; kiểm tra hai prediction CSV dùng cùng Test set và ground truth.", "100%", ""],
  ["2", "Chuẩn hóa metric và metadata", "Xuất Accuracy, macro Precision/Recall/F1, F1 từng lớp, confusion matrix, số mẫu, thiết bị và thông tin checkpoint.", "100%", ""],
  ["3", "Viết hướng dẫn đánh giá và phân tích", "Tạo `docs/NV5_EVALUATION.md`, quy định cột CSV, lệnh chạy và nguyên tắc không dùng Test để chọn model.", "100%", ""],
  ["4", "Chuẩn bị câu hỏi bảo vệ", "Tạo bộ 10 câu hỏi và đáp án về leakage, split, F1, Transfer Learning, giới hạn PlantVillage và phát hiện sớm.", "100%", ""],
  ["5", "Lập khung báo cáo Word", "Tạo báo cáo Word có các phần mục tiêu, dữ liệu, mô hình, đánh giá, kết luận và chỗ điền số liệu thật.", "100%", ""],
  ["6", "Lập khung slide thuyết trình", "Tạo slide editable về mục tiêu, dữ liệu, mô hình, giao thức đánh giá, kết quả, demo, giới hạn và hướng phát triển.", "100%", ""],
  ["7", "Tổng hợp kết quả cuối", "Đang chờ checkpoint, prediction CSV và metrics thật từ NV2/NV3 để sinh comparison.csv, confusion matrix và điền báo cáo.", "80%", ""],
];

let y = y0 + headerH;
for (let r = 0; r < rows.length; r++) {
  const fill = r % 2 === 0 ? lightBlue : lighter;
  x = x0;
  for (let c = 0; c < widths.length; c++) {
    shape("rect", x, y, widths[c], rowH, fill, "#FFFFFF");
    const align = c === 0 || c === 3 || c === 4 ? "center" : "left";
    const size = c === 2 ? 13.5 : c === 1 ? 14.5 : 14;
    text(rows[r][c], x + 7, y + 5, widths[c] - 14, rowH - 10, { size, color: blue, bold: c === 1, align });
    x += widths[c];
  }
  y += rowH;
}

text("Ghi chú: Kết quả tổng hợp cuối cùng chỉ được điền sau khi nhận đủ minh chứng mô hình và dữ liệu từ NV2, NV3, NV4.", 64, 642, 1140, 27, { size: 13, color: gray });

const png = await deck.export({ slide, format: "png", scale: 1 });
await fs.writeFile(preview, new Uint8Array(await png.arrayBuffer()));
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(output);
console.log(output);
