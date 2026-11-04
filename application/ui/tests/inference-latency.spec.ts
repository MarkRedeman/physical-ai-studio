import { HttpResponse } from 'msw';

import { getMockedDataset } from '../src/test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../src/test-utils/mocks/mock-environment';
import { getMockedInferenceMetrics } from '../src/test-utils/mocks/mock-inference-metrics';
import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { getMockedProject } from '../src/test-utils/mocks/mock-project';
import { getMockedRuntimeSession } from '../src/test-utils/mocks/mock-runtime-session';
import { getMockedSettings } from '../src/test-utils/mocks/mock-settings';
import { expect, http, test } from './fixtures';

test('shows live chunk latency and clears it when inference stops or restarts', async ({ page, network }, testInfo) => {
    network.use(
        http.get('/api/settings', () => HttpResponse.json(getMockedSettings())),
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.get('/api/projects/{project_id}', () =>
            HttpResponse.json(getMockedProject({ datasets: [getMockedDataset()] }))
        ),
        http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset())),
        http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([])),
        http.get('/api/projects/{project_id}/environments/{environment_id}', () =>
            HttpResponse.json(getMockedEnvironment())
        ),
        http.get('/api/projects/{project_id}/environments', () =>
            HttpResponse.json([{ id: 'environment-1', name: 'Workbench' }])
        ),
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        ),
        http.get('/api/models/{model_id}', () => HttpResponse.json({ model: getMockedModel(), exports: [] })),
        http.get('/api/projects/{project_id}/robots/{robot_id}', () =>
            HttpResponse.json({
                id: 'robot-1',
                name: 'Demo follower',
                type: 'demo-robot',
                unavailable: true,
                payload: {},
            })
        )
    );

    let emitSteadyPrediction = () => {};
    // No websocket is connected to real hardware in this preview.
    await page.routeWebSocket('**/api/**', (socket) => {
        if (!socket.url().includes('/runtime/ws')) return;
        const state = getMockedRuntimeSession({
            model: { model_id: 'model-1', name: 'Cube policy', backend: 'onnx', device: 'cpu' },
        });
        let run = 0;
        const publishState = () => socket.send(JSON.stringify({ type: 'state', data: state }));
        const publishTiming = (warmup: boolean) =>
            socket.send(
                JSON.stringify(
                    getMockedInferenceMetrics({
                        run_id: `00000000-0000-0000-0000-${String(run).padStart(12, '0')}`,
                        timestamp: Date.now() / 1000,
                        is_warmup: warmup,
                        latency_ms: warmup ? 284 : 42,
                        average_latency_ms: warmup ? null : 38,
                        sample_count: warmup ? 0 : 20,
                        prediction_count: warmup ? 1 : 21,
                    })
                )
            );
        emitSteadyPrediction = () => publishTiming(false);
        socket.onMessage((message) => {
            const command = JSON.parse(String(message));
            if (command.command === 'set_control') {
                if (command.control?.type === 'model') {
                    run += 1;
                    socket.send(JSON.stringify({ type: 'status', component: 'control', state: 'starting' }));
                    // Warm-up may finish before the new control's state reaches the UI.
                    publishTiming(true);
                } else {
                    socket.send(JSON.stringify({ type: 'status', component: 'control', state: 'stopped' }));
                }
                state.control = command.control;
            }
            publishState();
            socket.send(JSON.stringify({ type: 'ack', request_id: command.request_id, ok: true }));
        });
        publishState();
    });

    await page.setViewportSize({ width: 1440, height: 1050 });
    await page.goto('/projects/project-1/datasets/dataset-1/record');
    await page.getByRole('radio', { name: 'Inference', exact: true }).click();
    const footer = page.getByLabel('Inference performance');
    await expect(footer.getByText('Run the model to see latency')).toBeVisible();
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    await expect(footer.getByText('Warm-up')).toBeVisible();
    await expect(footer.getByText('284 ms')).toBeVisible();
    emitSteadyPrediction();
    await expect(footer.getByText('42 ms', { exact: true })).toBeVisible();
    await expect(footer.getByText('avg 38 ms')).toBeVisible();
    await footer.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath('inference-latency.png') });
    await page.getByRole('button', { name: 'Stop model', exact: true }).click();
    emitSteadyPrediction(); // A delayed result from the old run must not reappear.
    await expect(footer.getByText('Run the model to see latency')).toBeVisible();
    await expect(footer.getByText('42 ms', { exact: true })).toBeHidden();
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    await expect(footer.getByText('Warm-up')).toBeVisible();
    await expect(footer.getByText('avg 38 ms')).toBeHidden();
});
