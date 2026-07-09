import { Suspense, type ReactNode } from 'react';

import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
    render as rtlRender,
    renderHook as rtlRenderHook,
    type RenderOptions as RTLRenderOptions,
} from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

import { createQueryClient } from '../query-client/query-client';

type RenderOptions = RTLRenderOptions & {
    /** The URL the memory router starts at, e.g. '/projects/abc/environments/new'. */
    route?: string;
    /** The route pattern that matches `route`, e.g. '/projects/:project_id/environments/new'. */
    path?: string;
    /** Pass an existing QueryClient to share cache across multiple render calls in one test. */
    queryClient?: QueryClient;
};

const TestProviders = ({ children, queryClient }: { children: ReactNode; queryClient: QueryClient }) => (
    <QueryClientProvider client={queryClient}>
        <ThemeProvider>
            <Suspense>{children}</Suspense>
        </ThemeProvider>
    </QueryClientProvider>
);

const createTestRouter = (children: ReactNode, options: RenderOptions, queryClient: QueryClient) => {
    const route = options.route ?? '/';
    const path = options.path ?? '/';

    return createMemoryRouter(
        [
            {
                path,
                element: <TestProviders queryClient={queryClient}>{children}</TestProviders>,
            },
        ],
        { initialEntries: [route], initialIndex: 0 }
    );
};

export const render = (ui: ReactNode, options: RenderOptions = {}) => {
    const testQueryClient = options.queryClient ?? createQueryClient();
    const router = createTestRouter(ui, options, testQueryClient);

    return rtlRender(<RouterProvider router={router} />);
};

export const renderHook = <TProps, TResult>(callback: (props: TProps) => TResult, options: RenderOptions = {}) => {
    const testQueryClient = options.queryClient ?? createQueryClient();

    const Wrapper = ({ children }: { children: ReactNode }) => {
        const wrappedChildren = options.wrapper ? <options.wrapper>{children}</options.wrapper> : children;
        const router = createTestRouter(wrappedChildren, options, testQueryClient);

        return <RouterProvider router={router} />;
    };

    return rtlRenderHook(callback, { wrapper: Wrapper });
};
