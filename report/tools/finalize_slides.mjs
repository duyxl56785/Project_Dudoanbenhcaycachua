import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const skillDir = "C:/Users/PC/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations";
const candidate = path.join(root, "report", "deliverables", "slides_nhom_tomato_template.pptx");
const finalPath = path.join(root, "report", "deliverables", "slides_nhom_tomato_template_final.pptx");
const staging = path.join(root, ".codex-finalizer");
await fs.mkdir(staging, { recursive: true });
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
const result = await finalizePresentation({
  workspaceDir: root,
  candidatePath: candidate,
  finalPath,
  pythonExecutable: "C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
  requiredNativeTableOwnerSlides: [],
  fontPolicy: { basis: "design", families: ["Aptos", "Aptos Display"] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(staging, "slides_nhom_tomato_template.validation.json"),
});
console.log(JSON.stringify(result, null, 2));
