import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const skillDir = "C:/Users/PC/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations";
const candidate = path.join(root, ".report_build", "slides_nhom_tomato_two_models_nv5.pptx");
const finalPath = path.join(root, ".report_final_tmp", "slides_nhom_tomato_two_models_nv5.pptx");
const publishPath = path.join(root, "report", "deliverables", "slides_nhom_tomato_two_models_nv5.pptx");
const staging = path.join(root, ".codex-finalizer");
await fs.mkdir(staging, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
const result = await finalizePresentation({
  workspaceDir: root,
  candidatePath: candidate,
  finalPath,
  pythonExecutable: "C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit", "--require-native-table-slide", "6"],
  requiredNativeTableOwnerSlides: [6],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: "design", families: ["Aptos", "Aptos Display"] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(staging, "slides_nhom_tomato_two_models_nv5_v3.validation.json"),
});
await fs.copyFile(finalPath, publishPath);
console.log(JSON.stringify(result, null, 2));
