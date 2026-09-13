import type { Plugin } from "vite";
import { createPlacementHandler } from "../../../api/placement";

/** Server-only middleware: credentials are passed to the handler, never to Vite's client defines. */
export function placementApiPlugin(environment: { OPENAI_API_KEY?: string; OPEN_AI_KEY?: string; OPENAI_PLACEMENT_MODEL?: string } = {}): Plugin {
  const handler = createPlacementHandler({ apiKey: environment.OPENAI_API_KEY || environment.OPEN_AI_KEY, model: environment.OPENAI_PLACEMENT_MODEL });
  return {
    name: "holospex-placement-api", apply: "serve",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url?.split("?", 1)[0] !== "/api/placement") { next(); return; }
        void handler(request, response).catch(() => {
          if (!response.headersSent) {
            response.statusCode = 500;
            response.setHeader("Content-Type", "application/json");
            response.setHeader("Cache-Control", "no-store");
          }
          if (!response.writableEnded) response.end(JSON.stringify({ error: { code: "placement_failed", message: "Placement service is unavailable." } }));
        });
      });
    },
  };
}
