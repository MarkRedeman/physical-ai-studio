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
    await int8.getByRole('button', { name: 'Actions for OpenVINO INT8' }).click();
    await page.getByRole('menuitem', { name: 'Optimize', exact: true }).click();
    await expect(dialog.getByRole('button', { name: /Source export/ })).toContainText('OpenVINO · INT8');
    await expect(dialog.getByRole('status')).toContainText('Optimization pending');
});
