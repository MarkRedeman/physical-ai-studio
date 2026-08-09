import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import type { SchemaModel } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { ModelFormats } from './model-formats';

const projectId = 'b8b28d4f-e78f-48ad-afb8-03d060178a3c';
const modelId = '9340adfd-9632-4c54-8acd-8304f9dfda91';

const model = {
    id: modelId,
    project_id: projectId,
    name: 'Test model',
    policy: 'act',
    path: '/models/test',
    properties: {},
} as SchemaModel;

const mockModelDetail = (exports: Array<{ type: string; size_bytes: number; file_count: number }>) => {
    server.use(
        http.get('/api/models/{model_id}', () =>
            HttpResponse.json({
                model,
                exports,
                training_summary: null,
                hparams: null,
            })
        ),
        http.get('/api/policies/backends', () => HttpResponse.json({ act: ['torch', 'openvino'] }))
    );
};

const renderFormats = () =>
    render(<ModelFormats model={model} />, {
        route: `/projects/${projectId}/models`,
        path: '/projects/:project_id/models',
    });

describe('ModelFormats', () => {
    it('shows a Download button for available backends and no "Try to export"', async () => {
        mockModelDetail([
            { type: 'torch', size_bytes: 1024, file_count: 1 },
            { type: 'openvino', size_bytes: 2048, file_count: 2 },
        ]);

        renderFormats();

        expect(await screen.findByRole('button', { name: /download pytorch export/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /download openvino export/i })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /try to export/i })).not.toBeInTheDocument();
    });

    it('queues an export for the unavailable backend when clicking "Try to export"', async () => {
        const user = userEvent.setup();
        const payloads: Array<Record<string, unknown>> = [];
        mockModelDetail([{ type: 'torch', size_bytes: 1024, file_count: 1 }]);
        server.use(
            http.post('/api/models/{model_id}:export', async ({ request }) => {
                payloads.push((await request.json()) as Record<string, unknown>);
                return HttpResponse.json(
                    {
                        id: '04ac42d4-5581-47b6-a316-2db506949a19',
                        project_id: projectId,
                        type: 'model_export',
                        payload: { model_id: modelId, backends: ['openvino'], compress: true },
                        status: 'pending',
                        progress: 0,
                        message: 'Model export job submitted',
                    },
                    { status: 201 }
                );
            })
        );

        renderFormats();

        // OpenVINO export is missing for this model, so its card offers to export it.
        const tryExportButton = await screen.findByRole('button', { name: /try to export/i });
        await user.click(tryExportButton);

        await waitFor(() => expect(payloads).toHaveLength(1));
        expect(payloads[0]?.model_id).toBe(modelId);
        expect(payloads[0]?.backends).toEqual(['openvino']);
        expect(payloads[0]?.compress).toBe(true);
    });
});
