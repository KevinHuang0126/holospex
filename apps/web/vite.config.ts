import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { localSamplesPlugin } from "./dev/localSamples";
import { placementApiPlugin } from "./dev/placementApi";

export default defineConfig(({ mode }) => {
  // Read only private server settings from the repository's .env files and process environment.
  // Do not add these keys to envPrefix or define: neither belongs in the browser bundle.
  const privateEnvironment = loadEnv(mode, fileURLToPath(new URL("../../", import.meta.url)), ["OPENAI_API_KEY", "OPEN_AI_KEY", "OPENAI_PLACEMENT_MODEL"]);
  return {
    plugins: [react(), placementApiPlugin(privateEnvironment), ...(mode === "samples" ? [localSamplesPlugin(fileURLToPath(new URL("./tests/samples_tst", import.meta.url)))] : [])],
    server: { port: 5173 },
  };
});
