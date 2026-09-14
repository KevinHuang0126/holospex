import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { localTrainingPlugin, trainingReferenceDirectory } from "./dev/localTraining";
import { placementApiPlugin } from "./dev/placementApi";
import { identificationApiPlugin } from "./dev/identificationApi";

export default defineConfig(({ mode }) => {
  // Read only private server settings from the repository's .env files and process environment.
  // Do not add these keys to envPrefix or define: neither belongs in the browser bundle.
  const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
  const privateEnvironment = loadEnv(mode, repositoryRoot, ["OPENAI_API_KEY", "OPEN_AI_KEY", "OPENAI_PLACEMENT_MODEL", "HOLOSPEX_TRAINING_REFERENCE_DIR", "HOLOSPEX_IDENTIFICATION_URL", "HOLOSPEX_IDENTIFICATION_TOKEN"]);
  return {
    plugins: [react(), placementApiPlugin(privateEnvironment), identificationApiPlugin(privateEnvironment), ...(mode === "samples" ? [localTrainingPlugin(trainingReferenceDirectory(repositoryRoot, privateEnvironment.HOLOSPEX_TRAINING_REFERENCE_DIR))] : [])],
    server: { port: 5173 },
  };
});
