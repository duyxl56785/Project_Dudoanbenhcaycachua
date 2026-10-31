import fs from "node:fs/promises";
import path from "node:path";
import { createCanvas, loadImage } from "file:///C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas/index.js";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const sourceDir = path.join(root, "results");
const outputDir = path.join(root, "report", "assets", "results");

await fs.mkdir(outputDir, { recursive: true });
for (const stem of ["cnn_confusion_matrix", "resnet18_confusion_matrix"]) {
  const image = await loadImage(path.join(sourceDir, `${stem}.svg`));
  const canvas = createCanvas(image.width * 2, image.height * 2);
  const context = canvas.getContext("2d");
  context.fillStyle = "#FFFFFF";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  await fs.writeFile(path.join(outputDir, `${stem}.png`), canvas.toBuffer("image/png"));
}

console.log(outputDir);
