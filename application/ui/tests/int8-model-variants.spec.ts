import { HttpResponse } from 'msw';

import { getMockedExportVariant } from '../src/test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { getMockedModelDetailResponse } from '../src/test-utils/mocks/mock-model-detail-response';
import { getMockedOptimizeModelJob } from '../src/test-utils/mocks/mock-optimize-model-job';
import { expect, http, test } from './fixtures';

test('creates an INT8 export and selects it for download and inference', async ({ page, network }, testInfo) => {
    const model = getMockedModel({ available_backends: ['openvino'], dataset_id: null });
    const detail = getMockedModelDetailResponse({ model });
    let jobs: ReturnType<typeof getMockedOptimizeModelJob>[] = [];
    let complete = () => {};
    await page.routeWebSocket('**/api/jobs/ws', (socket) => {
        complete = () => {
            detail.exports[0].variants?.push(getMockedExportVariant({ id: 'int8', weight_compression: 'int8_sym' }));
            detail.available_actions = [];
            jobs = [getMockedOptimizeModelJob({ status: 'completed' })];
            socket.send(JSON.stringify({ event: 'JOB_UPDATE', data: jobs[0] }));
        };
    });
    await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
    network.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([detail.model])),
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/policies/backends', () => HttpResponse.json({ act: ['openvino'] })),
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.get('/api/system/devices/inference', () =>
            HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
        ),
        http.post('/api/models/{model_id}/optimize', () => {
            jobs = [getMockedOptimizeModelJob()];
            return HttpResponse.json(jobs[0], { status: 202 });
        })
    );
    await page.goto(`/projects/${model.project_id}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    await expect(page.getByRole('button', { name: 'Add variant' })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('available-model-cards.png'), fullPage: true });
    await page.getByRole('button', { name: 'Add variant' }).click();
    await page.getByRole('button', { name: 'Optimize to INT8' }).click();
    await expect(page.getByRole('dialog').getByRole('status')).toContainText('Export pending');
    complete();
    await page.getByRole('dialog').getByRole('button', { name: 'Close', exact: true }).click();
    const card = page.getByRole('region', { name: 'OpenVINO · INT8' });
    await expect(card).toBeVisible();
    await expect(page.getByRole('region', { name: 'OpenVINO · Original' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Add variant' })).toHaveCount(0);
    await card.getByRole('button', { name: 'Actions for OpenVINO INT8' }).click();
    await expect(page.getByRole('menuitem', { name: 'Download', exact: true })).toHaveAttribute('href', /variant=int8/);
    await page.keyboard.press('Escape');
    await page.screenshot({ path: testInfo.outputPath('int8-model-cards.png'), fullPage: true });
    await card.getByRole('button', { name: 'Run', exact: true }).click();
    await expect(page).toHaveURL(/inference\/openvino\?device=CPU&variant=int8$/);
});

test('adds a missing backend from a checkpoint without placeholder cards', async ({ page, network }, testInfo) => {
    const model = getMockedModel({ available_backends: [], dataset_id: null });
    const detail = getMockedModelDetailResponse({
        model,
        exports: [],
        available_actions: [{ backend: 'torch', operation: 'export' }],
    });
    const payload = {
        model_id: model.id!,
        backend: 'torch',
        operation: 'export',
        cancellation_requested: false,
    } as const;
    let jobs: ReturnType<typeof getMockedOptimizeModelJob>[] = [];
    let complete = () => {};
    await page.routeWebSocket('**/api/jobs/ws', (socket) => {
        complete = () => {
            detail.exports = [{ type: 'torch', size_bytes: 4096, file_count: 3, variants: [getMockedExportVariant()] }];
            detail.available_actions = [];
            detail.model = { ...model, available_backends: ['torch'] };
            jobs = [getMockedOptimizeModelJob({ payload, status: 'completed' })];
            socket.send(JSON.stringify({ event: 'JOB_UPDATE', data: jobs[0] }));
        };
    });
    await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
    network.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([detail.model])),
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.get('/api/system/devices/inference', () =>
            HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'torch', device: 'cpu' }])
        ),
        http.post('/api/models/{model_id}/exports/{backend}', () => {
            jobs = [getMockedOptimizeModelJob({ payload })];
            return HttpResponse.json(jobs[0], { status: 202 });
        })
    );
    await page.goto(`/projects/${model.project_id}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    await expect(page.getByText('Unavailable', { exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Add variant' }).click();
    await page.getByRole('dialog').getByRole('button', { name: 'Export model', exact: true }).click();
    await expect(page.getByRole('dialog').getByRole('status')).toContainText('Export pending');
    complete();
    await page.getByRole('dialog').getByRole('button', { name: 'Close', exact: true }).click();
    const card = page.getByRole('region', { name: 'PyTorch · Original' });
    await expect(card).toBeVisible();
    await expect(page.getByRole('heading', { name: 'OpenVINO', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Add variant' })).toHaveCount(0);
    await page.screenshot({ path: testInfo.outputPath('checkpoint-export-card.png'), fullPage: true });
    await card.getByRole('button', { name: 'Run', exact: true }).click();
    await expect(page).toHaveURL(/inference\/torch\?device=cpu&variant=original$/);
});
