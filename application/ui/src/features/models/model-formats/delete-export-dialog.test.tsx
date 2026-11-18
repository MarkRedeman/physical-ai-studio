import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { createQueryClient } from '../../../query-client/query-client';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { render } from '../../../test-utils/render';
import { BackendCard } from './backend-card';

const original = getMockedExportVariant({ size_bytes: 1024 ** 2 });
const int8 = getMockedExportVariant({ id: 'int8', size_bytes: 1024 ** 2 / 2 });

it.each(['export', 'backend'] as const)(
    'confirms the %s scope and refreshes model queries after removal',
    async (scope) => {
        const model = getMockedModel();
        const queryClient = createQueryClient();
        const detailKey = ['get', '/api/models/{model_id}', { params: { path: { model_id: model.id ?? '' } } }];
        const listKey = [
            'get',
            '/api/projects/{project_id}/models',
            { params: { path: { project_id: model.project_id } } },
        ];
        queryClient.setQueryData(detailKey, {});
        queryClient.setQueryData(listKey, []);
        let removed = false;
        server.use(
            http.get('/api/system/devices/inference', () => HttpResponse.json([])),
            http.delete('/api/models/{model_id}/exports/{backend}', () => {
                expect(scope).toBe('backend');
                removed = true;
                return new HttpResponse(null, { status: 204 });
            }),
            http.delete('/api/models/{model_id}/exports/{backend}/variants/{variant}', ({ params }) => {
                expect(scope).toBe('export');
                expect(params.variant).toBe('int8');
                removed = true;
                return new HttpResponse(null, { status: 204 });
            })
        );
        const user = userEvent.setup();
        render(
            <BackendCard
                model={model}
                backendType='openvino'
                variant={int8}
                variants={[original, int8]}
                onCreate={() => {}}
            />,
            { queryClient }
        );
        await user.click(screen.getByRole('button', { name: 'Actions for OpenVINO INT8' }));
        await user.click(await screen.findByRole('menuitem', { name: `Remove ${scope}…` }));
        const dialog = screen.getByRole('dialog');
        expect(removed).toBe(false);
        expect(
            within(dialog).getByText(/training checkpoint, training history, and other backends are kept/)
        ).toBeInTheDocument();
        expect(
            within(dialog).getByText(scope === 'backend' ? /Approximately 1.5 MB/ : /Approximately 0.5 MB/)
        ).toBeInTheDocument();
        if (scope === 'backend')
            expect(within(dialog).getByText('Exports removed: Original, INT8.')).toBeInTheDocument();
        await user.click(within(dialog).getByRole('button', { name: `Remove ${scope}` }));
        await waitFor(() => expect(removed).toBe(true));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(queryClient.getQueryState(detailKey)?.isInvalidated).toBe(true);
        expect(queryClient.getQueryState(listKey)?.isInvalidated).toBe(true);
    }
);

it('keeps the confirmation open on conflicts and allows canceling without deletion', async () => {
    let calls = 0;
    server.use(
        http.get('/api/system/devices/inference', () => HttpResponse.json([])),
        http.delete('/api/models/{model_id}/exports/{backend}/variants/{variant}', () => {
            calls++;
            return HttpResponse.json(
                { detail: [{ loc: [], msg: 'Unload the model first.', type: 'conflict' }] },
                { status: 409 }
            );
        })
    );
    const user = userEvent.setup();
    render(<BackendCard model={getMockedModel()} backendType='openvino' variant={original} onCreate={() => {}} />);
    await user.click(screen.getByRole('button', { name: 'Actions for OpenVINO Original' }));
    await user.click(await screen.findByRole('menuitem', { name: 'Remove export…' }));
    await user.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Remove export' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Unload the model first.');
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(calls).toBe(1);
});
