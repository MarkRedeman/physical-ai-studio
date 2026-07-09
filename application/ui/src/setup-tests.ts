import '@testing-library/jest-dom';

import { afterAll, afterEach, beforeAll } from 'vitest';

import { server } from './msw-node-setup';

process.env.PUBLIC_API_BASE_URL = 'http://localhost:7860';

// Start MSW at module-evaluation time so that globalThis.fetch is patched before
// any test-file import (e.g. src/api/client.ts) captures it via
//   `fetch: baseFetch = globalThis.fetch`
// in openapi-fetch.  If we defer to beforeAll, client.ts is imported first and
// keeps a stale reference to the pre-patch fetch.
server.listen({ onUnhandledRequest: 'bypass' });

beforeAll(() => {
    // server.listen() is already called above; this is a no-op guard in case
    // vitest ever re-evaluates setup files between test files.
});

afterEach(() => {
    server.resetHandlers();
});

afterAll(() => {
    server.close();
});
