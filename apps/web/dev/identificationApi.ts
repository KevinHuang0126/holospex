import type { Plugin } from "vite";
import { createIdentificationHandler, IDENTIFICATION_UNAVAILABLE } from "../../../api/identify";

/** These settings are read by the development server, never included in client defines. */
export function identificationApiPlugin(environment: { HOLOSPEX_IDENTIFICATION_URL?: string; HOLOSPEX_IDENTIFICATION_TOKEN?: string } = {}): Plugin {
  const handler = createIdentificationHandler({ endpoint: environment.HOLOSPEX_IDENTIFICATION_URL || "http://127.0.0.1:8765/identify", token: environment.HOLOSPEX_IDENTIFICATION_TOKEN });
  return {
    name: "holospex-identification-api", apply: "serve",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url?.split("?", 1)[0] !== "/api/identify") { next(); return; }
        void handler(request, response).catch(() => {
          if (!response.headersSent) {
            response.statusCode = 503; response.setHeader("Content-Type", "application/json"); response.setHeader("Cache-Control", "no-store");
          }
          if (!response.writableEnded) response.end(JSON.stringify({ status: "unavailable", message: IDENTIFICATION_UNAVAILABLE }));
        });
      });
    },
  };
}
