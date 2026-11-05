// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpResponse } from 'msw';

import { getMockedExportVariant } from '../src/test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { getMockedModelDetailResponse } from '../src/test-utils/mocks/mock-model-detail-response';
import { getMockedOptimizeModelJob } from '../src/test-utils/mocks/mock-optimize-model-job';
import { expect, http, test } from './fixtures';

test('model rows show separate export cards and create variants through the empty card', async ({
    page,
    network,
}, testInfo) => {
    const model = getMockedModel({ dataset_id: null, policy: 'smolvla' });
    const detail = getMockedModelDetailResponse({
        model,
        exports: [
            {
                type: 'openvino',
                size_bytes: 1048576,
                file_count: 2,
                variants: [
                    getMockedExportVariant({
                        available_operations: [
                            {
                                operation: 'int8_sym',
                                target_variant: 'int8',
                                available: false,
                                reason: 'This variant already exists',
                            },
                        ],
                    }),
                    getMockedExportVariant({
                        id: 'int8',
                        label: 'INT8',
                        weight_compression: 'int8_sym',
                        available_operations: [{ operation: 'rtc', target_variant: 'rtc-int8', available: true }],
                    }),
                ],
            },
        ],
    });
    const job = getMockedOptimizeModelJob({
        payload: {
            ...getMockedOptimizeModelJob().payload,
            source_variant: 'int8',
            operation: 'rtc',
            target_variant: 'rtc-int8',
        },
    });
    let submitted: unknown;
    network.use(
        http.get('/api/system/devices/inference', () =>
            HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
        ),
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([model])),
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.post('/api/models/{model_id}/exports/{backend}/variants', async ({ params, request }) => {
            submitted = { backend: params.backend, body: await request.json() };
            network.use(http.get('/api/jobs', () => HttpResponse.json([job])));
            return HttpResponse.json(job, { status: 202 });
        })
    );

    await page.goto(`/projects/${model.project_id}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    const original = page.getByRole('region', { name: 'OpenVINO · Original' });
    const int8 = page.getByRole('region', { name: 'OpenVINO · INT8' });
    await expect(original).toBeVisible();
    await expect(int8).toBeVisible();
    const backendName = int8.getByRole('heading', { name: 'OpenVINO', exact: true });
    const description = int8.getByText('INT8', { exact: true });
    await expect(original.getByRole('heading', { name: 'OpenVINO', exact: true }).locator('..')).toHaveText('OpenVINO');
    const nameStyle = await backendName.evaluate((element) => ({
        fontSize: getComputedStyle(element).fontSize,
        color: getComputedStyle(element).color,
    }));
    const descriptionStyle = await description.evaluate((element) => ({
        fontSize: getComputedStyle(element).fontSize,
        color: getComputedStyle(element).color,
    }));
    expect(Number.parseFloat(descriptionStyle.fontSize)).toBeLessThan(Number.parseFloat(nameStyle.fontSize));
    expect(descriptionStyle.color).not.toBe(nameStyle.color);
    const nameBox = await backendName.boundingBox();
    const descriptionBox = await description.boundingBox();
    if (!nameBox || !descriptionBox) throw new Error('Expected backend name and description to be visible');
    expect(descriptionBox.x).toBeGreaterThan(nameBox.x + nameBox.width);
    expect(Math.abs(descriptionBox.y + descriptionBox.height / 2 - nameBox.y - nameBox.height / 2)).toBeLessThan(4);
    const weights = await int8.getByText('Weights', { exact: true }).locator('..').boundingBox();
    const optimize = await int8.getByRole('button', { name: 'Optimize', exact: true }).boundingBox();
    const run = await int8.getByRole('button', { name: 'Run', exact: true }).boundingBox();
    if (!weights || !optimize || !run) throw new Error('Expected weights, Optimize and Run to be visible');
    expect(optimize.x).toBeGreaterThan(weights.x + weights.width);
    expect(run.x).toBeGreaterThan(optimize.x + optimize.width);
    expect(Math.abs(run.y + run.height - weights.y - weights.height)).toBeLessThan(3);
    await expect(original.getByRole('button', { name: 'Optimize', exact: true })).toHaveCount(0);
    await original.getByRole('button', { name: 'Actions for OpenVINO Original' }).click();
    const originalMenu = page.getByRole('menu', { name: 'Actions for OpenVINO Original' });
    await expect(originalMenu.getByRole('menuitem', { name: 'Download', exact: true })).toHaveAttribute(
        'href',
        /variant=original/
    );
    await expect(originalMenu.getByRole('menuitem', { name: 'Optimize', exact: true })).toHaveCount(0);
    await page.keyboard.press('Escape');
    await expect(originalMenu).toBeHidden();
    await int8.getByRole('button', { name: 'Actions for OpenVINO INT8' }).click();
    const int8Menu = page.getByRole('menu', { name: 'Actions for OpenVINO INT8' });
    await expect(int8Menu.getByRole('menuitem', { name: 'Download', exact: true })).toHaveAttribute(
        'href',
        /variant=int8/
    );
    await page.keyboard.press('Escape');
    await expect(int8Menu).toBeHidden();
    await expect(page.getByRole('button', { name: 'Optimize to INT8' })).toHaveCount(0);
    await original.locator('..').screenshot({ path: testInfo.outputPath('variant-cards.png') });

    await page.getByRole('button', { name: 'Add variant' }).click();
    const dialog = page.getByRole('dialog', { name: 'Create model variant' });
    await expect(dialog.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
    await expect(dialog.getByText('This variant already exists')).toBeVisible();
    await dialog.getByRole('button', { name: /Source export/ }).click();
    await page.getByRole('option', { name: 'OpenVINO · INT8' }).click();
    await dialog.getByRole('button', { name: 'Export RTC' }).click();
    await expect
        .poll(() => submitted)
        .toEqual({ backend: 'openvino', body: { source_variant: 'int8', operation: 'rtc' } });
    await expect(dialog.getByRole('status')).toContainText('Optimization pending');
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    await expect(dialog).toBeHidden();
    await expect(page.getByRole('status')).toHaveText('1 variant job in progress');
    await int8.getByRole('button', { name: 'Optimize', exact: true }).click();
    await expect(dialog.getByRole('button', { name: /Source export/ })).toContainText('OpenVINO · INT8');
    await expect(dialog.getByRole('status')).toContainText('Optimization pending');
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    await expect(dialog).toBeHidden();
    // Keep this navigation test isolated from physical hardware and real sessions.
    await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
    await int8.getByRole('button', { name: 'Run', exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`/models/${model.id}/inference/openvino\\?device=CPU&variant=int8$`));
    await expect(page.getByText(`${model.name} · openvino · int8`)).toBeVisible();
});

test('PyTorch variants can run directly on a Torch device', async ({ page, network }) => {
    const model = getMockedModel({ dataset_id: null, policy: 'smolvla', available_backends: ['torch'] });
    const detail = getMockedModelDetailResponse({
        model,
        exports: [
            {
                type: 'torch',
                size_bytes: 1048576,
                file_count: 2,
                variants: [
                    getMockedExportVariant({ can_run_in_studio: true }),
                    getMockedExportVariant({
                        id: 'snapflow',
                        label: 'SnapFlow',
                        checkpoint_kind: 'snapflow',
                        can_run_in_studio: true,
                    }),
                ],
            },
        ],
    });
    network.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([model])),
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.get('/api/system/devices/inference', () =>
            HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'torch', device: 'cpu' }])
        )
    );
    await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
    await page.goto(`/projects/${model.project_id}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    const original = page.getByRole('region', { name: 'PyTorch · Original' });
    const distilled = page.getByRole('region', { name: 'PyTorch · SnapFlow' });
    await expect(original.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
    await expect(distilled.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
    await distilled.getByRole('button', { name: 'Run', exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`/models/${model.id}/inference/torch\\?device=cpu&variant=snapflow$`));
    await expect(page.getByText(`${model.name} · torch · snapflow`)).toBeVisible();
});
