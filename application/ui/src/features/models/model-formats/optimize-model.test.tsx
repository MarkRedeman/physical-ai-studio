import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { createQueryClient } from '../../../query-client/query-client';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedOptimizeModelJob } from '../../../test-utils/mocks/mock-optimize-model-job';
import { render } from '../../../test-utils/render';
import { JOBS_QUERY_KEY } from '../../jobs/use-job-cache';
import { OptimizeModel } from './optimize-model';

const renderOptimizer = (variant = getMockedExportVariant()) => {
    const queryClient = createQueryClient();
    render(
        <OptimizeModel
            modelId='model-1'
            backend='openvino'
            variant={variant.id}
            operations={variant.available_operations}
        />,
        { queryClient }
    );
    return queryClient;
};

describe('OptimizeModel', () => {
    beforeEach(() => server.use(http.get('/api/jobs', () => HttpResponse.json([]))));

    it('submits once with the selected source and caches the queued job', async () => {
        const user = userEvent.setup();
        const job = getMockedOptimizeModelJob();
        const submit = rs.fn();
        server.use(
            http.post('/api/models/{model_id}/exports/{backend}/variants', async ({ params, request }) => {
                submit(params, await request.json());
                server.use(http.get('/api/jobs', () => HttpResponse.json([job])));
                return HttpResponse.json(job, { status: 202 });
            })
        );
        const client = renderOptimizer();
        const button = screen.getByRole('button', { name: 'Optimize to INT8' });
        expect(button).toBeDisabled();
        await waitFor(() => expect(button).toBeEnabled());
        await user.click(button);
        expect(await screen.findByRole('status')).toHaveTextContent('Optimization pending');
        await user.click(button);
        expect(submit).toHaveBeenCalledTimes(1);
        expect(submit).toHaveBeenCalledWith(
            { model_id: 'model-1', backend: 'openvino' },
            { source_variant: 'original', operation: 'int8_sym' }
        );
        expect(client.getQueryData(JOBS_QUERY_KEY)).toEqual([job]);
    });

    it('surfaces submission errors and permits retry', async () => {
        server.use(
            http.post('/api/models/{model_id}/exports/{backend}/variants', ({ response }) =>
                response.untyped(HttpResponse.json({ message: 'Not enough disk space' }, { status: 409 }))
            )
        );
        const user = userEvent.setup();
        renderOptimizer();
        const button = screen.getByRole('button', { name: 'Optimize to INT8' });
        await waitFor(() => expect(button).toBeEnabled());
        await user.click(button);
        expect(await screen.findByRole('alert')).toHaveTextContent('Not enough disk space');
        expect(button).toBeEnabled();
    });

    it.each(['pending', 'running'] as const)('shows %s progress and can cancel the job', async (status) => {
        const job = getMockedOptimizeModelJob({ status, progress: 42 });
        const cancel = rs.fn();
        server.use(
            http.get('/api/jobs', () => HttpResponse.json([job])),
            http.post('/api/jobs/{job_id}:interrupt', ({ params }) => {
                cancel(params.job_id);
                server.use(http.get('/api/jobs', () => HttpResponse.json([{ ...job, status: 'canceled' }])));
                return HttpResponse.json(null);
            })
        );
        const user = userEvent.setup();
        renderOptimizer();
        expect(await screen.findByRole('progressbar', { name: 'Optimization progress' })).toHaveAttribute(
            'aria-valuenow',
            '42'
        );
        expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
        await user.click(screen.getByRole('button', { name: 'Cancel optimization' }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeEnabled());
        expect(cancel).toHaveBeenCalledWith(job.id);
    });

    it('does not confuse other backends or source variants with the current source', async () => {
        const job = getMockedOptimizeModelJob({ status: 'running' });
        server.use(
            http.get('/api/jobs', () =>
                HttpResponse.json([
                    { ...job, payload: { ...job.payload, source_variant: 'rtc' } },
                    { ...job, id: 'onnx-job', payload: { ...job.payload, backend: 'onnx' } },
                ])
            )
        );
        renderOptimizer();
        await waitFor(() => expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeEnabled());
        expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });

    it('shows the latest matching failure and allows a new attempt', async () => {
        const failed = getMockedOptimizeModelJob({
            status: 'failed',
            message: 'Not enough memory',
            created_at: '2026-11-05T10:00:00Z',
        });
        server.use(
            http.get('/api/jobs', () =>
                HttpResponse.json([
                    failed,
                    getMockedOptimizeModelJob({ id: 'old', status: 'completed', created_at: '2026-11-04T10:00:00Z' }),
                    getMockedOptimizeModelJob({
                        id: 'other',
                        status: 'running',
                        payload: { ...failed.payload, model_id: 'another-model' },
                        created_at: '2026-11-05T11:00:00Z',
                    }),
                ])
            )
        );
        renderOptimizer();
        expect(await screen.findByRole('status')).toHaveTextContent('Optimization failed: Not enough memory');
        expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeEnabled();
    });

    it('surfaces cancellation errors without allowing duplicate submissions', async () => {
        const user = userEvent.setup();
        const job = getMockedOptimizeModelJob({ status: 'running' });
        server.use(
            http.get('/api/jobs', () => HttpResponse.json([job])),
            http.post('/api/jobs/{job_id}:interrupt', ({ response }) =>
                response.untyped(HttpResponse.json({ message: 'Cancellation unavailable' }, { status: 409 }))
            )
        );
        renderOptimizer();
        await user.click(await screen.findByRole('button', { name: 'Cancel optimization' }));
        expect(await screen.findByRole('alert')).toHaveTextContent('Cancellation unavailable');
        expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Cancel optimization' })).toBeEnabled();
    });

    it('explains unavailable operations without allowing submission', async () => {
        renderOptimizer(
            getMockedExportVariant({
                available_operations: [
                    { operation: 'rtc', available: false, target_variant: 'rtc', reason: 'RTC requires a checkpoint' },
                ],
            })
        );
        expect(screen.getByRole('button', { name: 'Export RTC' })).toBeDisabled();
        expect(screen.getByText('RTC requires a checkpoint')).toBeInTheDocument();
    });

    it('blocks submission when job state cannot be loaded and offers retry', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.error()));
        const user = userEvent.setup();
        renderOptimizer();
        expect(await screen.findByRole('alert')).toHaveTextContent('Could not load optimization status');
        expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));
        await user.click(screen.getByRole('button', { name: 'Retry status' }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeEnabled());
    });
});
