import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "report", "bao_cao_cong_viec_nv4.pptx");
const preview = path.join(root, ".nv4_slide_preview.png");
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slide = deck.slides.add();
const blue = "#2F66B3", headerBlue = "#1388C5", lightBlue = "#DCE6F2", lighter = "#F0F4FA", dark = "#1F4E79", red = "#B83A2E", gray = "#4A5568";
slide.background.fill = "#FFFFFF";

function shape(geometry, x, y, w, h, fill, lineFill = fill) {
  return slide.shapes.add({ geometry, position: { left: x, top: y, width: w, height: h }, fill, line: { style: "solid", fill: lineFill, width: 1 } });
}
function text(value, x, y, w, h, options = {}) {
  const box = slide.shapes.add({ geometry: "textbox", position: { left: x, top: y, width: w, height: h }, fill: "none", line: { style: "solid", fill: "none", width: 0 } });
  box.text = value;
  box.text.style = { fontFamily: "Aptos", fontSize: options.size ?? 17, color: options.color ?? dark, bold: options.bold ?? false, align: options.align ?? "left", verticalAlign: "middle" };
  return box;
}

shape("rect", 0, 0, 1280, 58, headerBlue, headerBlue);
text("BÁO CÁO CÔNG VIỆC: NHẬN DIỆN BỆNH QUA ẢNH LÁ CÀ CHUA", 44, 8, 1050, 40, { size: 25, color: "#FFFFFF", bold: true, align: "center" });
text("Sinh viên: Nguyễn Minh Duy", 62, 77, 560, 32, { size: 22, color: red, bold: true });
text("Nhiệm vụ NV4 – Demo Streamlit và tích hợp mô hình", 62, 112, 800, 26, { size: 16, color: gray });
text("Nhóm 4 · Học máy và Khai phá dữ liệu", 920, 93, 290, 26, { size: 15, color: blue, bold: true, align: "right" });

const x0 = 62, y0 = 160, widths = [58, 252, 540, 130, 190], headerH = 58, rowH = 58;
const headers = ["STT", "NỘI DUNG CÔNG VIỆC", "KẾT QUẢ", "TỰ ĐÁNH GIÁ", "TRƯỞNG NHÓM\nĐÁNH GIÁ"];
let x = x0;
for (let i = 0; i < widths.length; i++) {
  shape("rect", x, y0, widths[i], headerH, blue, "#FFFFFF");
  text(headers[i], x + 5, y0 + 6, widths[i] - 10, headerH - 12, { size: 15, color: "#FFFFFF", bold: true, align: "center" });
  x += widths[i];
}

const rows = [
  ["1", "Thiết kế giao diện Streamlit", "Hoàn thiện app upload ảnh JPG/PNG, xem trước ảnh, chọn CNN, ResNet18 hoặc so sánh cả hai.", "100%", ""],
  ["2", "Xây API suy luận dùng chung", "Hoàn thiện `src/inference.py`: đọc mapping, config, preprocessing, load model và trả top-1 cùng xác suất 4 lớp.", "100%", ""],
  ["3", "Tích hợp xử lý lỗi và giới hạn", "Kiểm tra ảnh hỏng, định dạng sai, file trên 10 MiB, thiếu checkpoint, CUDA và hiển thị cảnh báo không thay thế chuyên gia.", "100%", ""],
  ["4", "Đo thời gian và cache model", "Đo forward pass riêng, đồng bộ CUDA khi cần và cache model; không tính thời gian tải model/đọc ảnh vào số đo.", "100%", ""],
  ["5", "Kiểm thử giao diện và server", "Smoke test Streamlit không có exception; health check trả `ok`; kiểm tra cú pháp và bộ test inference đã có kết quả ghi nhận.", "90%", ""],
  ["6", "Tích hợp checkpoint thật", "Chưa nghiệm thu Gate 4 vì còn chờ checkpoint đã huấn luyện, PR phụ thuộc và định dạng state_dict thống nhất từ NV2/NV3.", "70%", ""],
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
text("Ghi chú: giao diện và API đã có; kết quả dự đoán end-to-end chỉ được xác nhận sau khi nhận đủ hai checkpoint thật.", 64, 594, 1140, 28, { size: 13, color: gray });
text("Nguồn kiểm tra: docs/DEMO_README.md trên nhánh feature/nv4-demo.", 64, 642, 1140, 25, { size: 12, color: gray });

const png = await deck.export({ slide, format: "png", scale: 1 });
await fs.writeFile(preview, new Uint8Array(await png.arrayBuffer()));
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(output);
console.log(output);
