import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useLocation } from 'react-router';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { getMockedModelDetailResponse } from '../../../test-utils/mocks/mock-model-detail-response';
import { render } from '../../../test-utils/render';
import { StartInferenceDialog } from './start-inference-dialog';

const CurrentLocation = () => {
    const { pathname, search } = useLocation();
    return <output aria-label='Current location'>{pathname + search}</output>;
};

describe('StartInferenceDialog model selection', () => {
    it('carries the selected variant into the inference URL', async () => {
        server.use(
            http.get('/api/models/{model_id}', () =>
                HttpResponse.json(
                    getMockedModelDetailResponse({
                        exports: [
                            {
                                type: 'openvino',
                                size_bytes: 1,
                                file_count: 1,
                                variants: [
                                    getMockedExportVariant(),
                                    getMockedExportVariant({ id: 'rtc-int8', label: 'RTC + INT8', rtc_enabled: true }),
                                ],
                            },
                        ],
                    })
                )
            ),
            http.get('/api/policies/backends', () => HttpResponse.json({ act: ['openvino'] })),
            http.get('/api/system/devices/inference', () =>
                HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
            )
        );
        const user = userEvent.setup();
        render(
            <>
                <StartInferenceDialog model={getMockedModel()} close={rs.fn()} />
                <CurrentLocation />
            </>,
            { path: '*', route: '/projects/project-1/models' }
        );
        await user.click(await screen.findByRole('button', { name: /Variant/ }));
        await user.click(await screen.findByRole('option', { name: 'RTC + INT8' }));
        await user.click(screen.getByRole('button', { name: 'Start' }));
        expect(screen.getByLabelText('Current location')).toHaveTextContent('variant=rtc-int8');
    });
    it.each([
        getMockedModel(),
        getMockedModel({
            id: 'optimized-model',
            name: 'pick-and-place (INT8)',
            train_job_id: null,
            optimization: { source_model_id: 'model-1', mode: 'int8_sym' },
        }),
    ])('runs the selected row using its own model id ($id) and the ordinary OpenVINO backend', async (model) => {
        server.use(
            http.get('/api/models/{model_id}', () => HttpResponse.json(getMockedModelDetailResponse({ model }))),
            http.get('/api/policies/backends', () => HttpResponse.json({ act: ['openvino', 'torch'] })),
            http.get('/api/system/devices/inference', () =>
                HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
            )
        );
        const close = rs.fn();
        const user = userEvent.setup();
        render(
            <>
                <StartInferenceDialog model={model} close={close} />
                <CurrentLocation />
            </>,
            { path: '*', route: '/projects/project-1/models' }
        );
        await waitFor(() => expect(screen.getByRole('button', { name: 'Start' })).toBeEnabled());
        await user.click(screen.getByRole('button', { name: 'Start' }));

        expect(screen.getByLabelText('Current location')).toHaveTextContent(
            `/projects/project-1/models/${model.id}/inference/openvino?device=CPU`
        );
        expect(close).toHaveBeenCalledTimes(1);
    });
});
