import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';
import { defineConfig } from 'vitest/config';

export default defineConfig({
    plugins: [
        react(),
        svgr({
            svgrOptions: {
                svgo: false,
                exportType: 'named',
            },
            include: '**/*.svg',
        }),
    ],
    test: {
        environment: 'jsdom',
        environmentOptions: {
            jsdom: {
                // Match the base URL that MSW handlers are registered under so that
                // relative fetch calls (baseUrl: '') resolve to the same origin.
                url: 'http://localhost:7860/',
            },
        },
        // This is needed to use globals like describe or expect
        globals: true,
        include: ['./src/**/*.test.{ts,tsx}'],
        setupFiles: './src/setup-tests.ts',
        watch: false,
        server: {
            deps: {
                inline: [/@react-spectrum\/.*/, /@spectrum-icons\/.*/, /@adobe\/react-spectrum\/.*/, /@geti-ui\/.*/],
            },
        },
    },
});
