import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModelDetailResponse } from '../../../test-utils/mocks/mock-model-detail-response';
import { getMockedOptimizeModelJob } from '../../../test-utils/mocks/mock-optimize-model-job';
import { render } from '../../../test-utils/render';
import { ModelFormats } from './model-formats';

it('shows only published backend/variant cards with no unavailable placeholders', async () => {
    const detail = getMockedModelDetailResponse({ available_actions: [] });
    detail.exports[0].variants = [getMockedExportVariant(), getMockedExportVariant({ id: 'int8' })];
    detail.exports.push({ type: 'torch', size_bytes: 4096, file_count: 3, variants: [getMockedExportVariant()] });
    server.use(
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/system/devices/inference', () => HttpResponse.json([]))
    );
    render(<ModelFormats model={detail.model} />, { path: '*', route: '/projects/project-1/models' });
    expect(await screen.findByRole('region', { name: 'OpenVINO · Original' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'OpenVINO · INT8' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'PyTorch · Original' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'ONNX' })).not.toBeInTheDocument();
    expect(screen.queryByText('Unavailable')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Add variant' })).not.toBeInTheDocument();
});

it('offers a checkpoint export even when no backend has been published', async () => {
    const detail = getMockedModelDetailResponse({
        exports: [],
        available_actions: [{ backend: 'torch', operation: 'export' }],
    });
    let submitted = false;
    const job = getMockedOptimizeModelJob({
        payload: {
            model_id: 'model-1',
            backend: 'torch',
            operation: 'export',
            cancellation_requested: false,
        },
    });
    server.use(
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/jobs', () => HttpResponse.json(submitted ? [job] : [])),
        http.post('/api/models/{model_id}/exports/{backend}', ({ params }) => {
            expect(params.backend).toBe('torch');
            submitted = true;
            return HttpResponse.json(job, { status: 202 });
        })
    );
    const user = userEvent.setup();
    render(<ModelFormats model={detail.model} />);
    await user.click(await screen.findByRole('button', { name: 'Add variant' }));
    const dialog = screen.getByRole('dialog', { name: 'Create model variant' });
    const start = within(dialog).getByRole('button', { name: 'Export model' });
    await waitFor(() => expect(start).toBeEnabled());
    await user.click(start);
    await waitFor(() => expect(submitted).toBe(true));
    expect(await within(dialog).findByRole('status')).toHaveTextContent('Export pending');
    expect(screen.queryByRole('region', { name: 'PyTorch · Original' })).not.toBeInTheDocument();
});

it('opens the INT8 action from the original card', async () => {
    const detail = getMockedModelDetailResponse();
    server.use(
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.get('/api/system/devices/inference', () => HttpResponse.json([]))
    );
    const user = userEvent.setup();
    render(<ModelFormats model={detail.model} />, { path: '*', route: '/projects/project-1/models' });
    const card = await screen.findByRole('region', { name: 'OpenVINO · Original' });
    await user.click(within(card).getByRole('button', { name: 'Optimize' }));
    expect(within(screen.getByRole('dialog')).getByRole('button', { name: 'Optimize to INT8' })).toBeInTheDocument();
});
