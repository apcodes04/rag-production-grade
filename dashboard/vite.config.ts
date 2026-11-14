import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Tailwind v4 runs as a Vite plugin, so there is no postcss.config.js and no
// tailwind.config.js: the theme lives in src/index.css and content is detected
// automatically.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173 },
});
