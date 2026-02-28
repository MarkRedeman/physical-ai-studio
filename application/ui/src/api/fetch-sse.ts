// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from './client';

/**
 * Construct a full URL for non-fetchClient calls (EventSource, img, etc.).
 *
 * When API_BASE_URL is '/' (the default), this simply returns the path
 * unchanged so that the browser resolves it against the current origin
 * (which the dev-server proxy will forward to the backend).
 */
const getApiUrl = (path: string): string => {
    const base: string = API_BASE_URL;
    if (!base || base === '/') {
        return path;
    }
    try {
        return new URL(path, base).toString();
    } catch {
        const normalizedBase = base.replace(/\/+$/, '');
        const normalizedPath = path.replace(/^\/+/, '');
        return `${normalizedBase}/${normalizedPath}`;
    }
};

/**
 * Connect to an SSE endpoint and yield parsed messages as an async iterable.
 *
 * Uses the browser-native EventSource API.  The iterable completes when the
 * server sends a `"DONE"` or `"COMPLETED"` data payload, or when the
 * connection errors out.
 */
export function fetchSSE<T = unknown>(url: string) {
    return {
        async *[Symbol.asyncIterator](): AsyncGenerator<T> {
            const eventSource = new EventSource(getApiUrl(url));

            try {
                let { promise, resolve, reject } = Promise.withResolvers<string>();

                eventSource.onmessage = (event) => {
                    if (event.data === 'DONE' || event.data.includes('COMPLETED')) {
                        eventSource.close();
                        resolve('DONE');
                        return;
                    }
                    resolve(event.data);
                };

                eventSource.onerror = () => {
                    eventSource.close();
                    reject(new Error('EventSource connection failed'));
                };

                while (true) {
                    const message = await promise;

                    if (message === 'DONE') {
                        break;
                    }

                    try {
                        yield JSON.parse(message);
                    } catch {
                        // Skip unparseable messages
                    }

                    ({ promise, resolve, reject } = Promise.withResolvers<string>());
                }
            } finally {
                eventSource.close();
            }
        },
    };
}
