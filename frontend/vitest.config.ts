import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Test-only config, separate from vite.config.ts / tsconfig.app.json build path.
// Keeps `npm run build` (tsc -b && vite build) decoupled from test tooling.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.tsx', 'src/**/*.test.ts'],
    setupFiles: ['src/test-setup.ts'],
    globals: true,
  },
})
