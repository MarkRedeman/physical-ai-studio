import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { getMockedModelDetailResponse } from '../../../test-utils/mocks/mock-model-detail-response';
import { getMockedOptimizeModelJob } from '../../../test-utils/mocks/mock-optimize-model-job';
import { render } from '../../../test-utils/render';
import { ModelFormats } from './model-formats';

const detail = getMockedModelDetailResponse({
    exports: [
        {
            type: 'openvino',
            size_bytes: 1048576,
            file_count: 2,
            variants: [
                getMockedExportVariant(),
                getMockedExportVariant({
                    id: 'int8',
                    label: 'INT8',
                    weight_compression: 'int8_sym',
                    available_operations: [{ operation: 'rtc', target_variant: 'rtc-int8', available: true }],
                }),
            ],
        },
        { type: 'onnx', size_bytes: 1048576, file_count: 2, variants: [getMockedExportVariant()] },
    ],
});

describe('ModelFormats variant cards', () => {
    beforeEach(() =>
        server.use(
            http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
            http.get('/api/jobs', () => HttpResponse.json([]))
        )
    );

    it('renders each backend variant as its own card and keeps creation controls in a dialog', async () => {
        const user = userEvent.setup();
        render(<ModelFormats model={getMockedModel()} />);
        expect(await screen.findByRole('region', { name: 'OpenVINO · Original' })).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'OpenVINO · INT8' })).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'ONNX · Original' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Optimize to INT8' })).not.toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Add variant' }));
        const dialog = within(await screen.findByRole('dialog', { name: 'Create model variant' }));
        expect(dialog.getByRole('button', { name: /Source export/ })).toBeInTheDocument();
        expect(dialog.getByRole('button', { name: 'Optimize to INT8' })).toBeInTheDocument();
        await user.click(dialog.getByRole('button', { name: 'Close' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });

    it('creates a derivative of the selected source, then shows its active job outside the dialog', async () => {
        const user = userEvent.setup();
        const job = getMockedOptimizeModelJob({
            payload: {
                ...getMockedOptimizeModelJob().payload,
                source_variant: 'int8',
                target_variant: 'rtc-int8',
                operation: 'rtc',
            },
        });
        server.use(
            http.post('/api/models/{model_id}/exports/{backend}/variants', async ({ params, request }) => {
                expect(params.backend).toBe('openvino');
                expect(await request.json()).toEqual({ source_variant: 'int8', operation: 'rtc' });
                server.use(http.get('/api/jobs', () => HttpResponse.json([job])));
                return HttpResponse.json(job, { status: 202 });
            })
        );
        render(<ModelFormats model={getMockedModel()} />);
        await user.click(await screen.findByRole('button', { name: 'Add variant' }));
        const dialog = within(screen.getByRole('dialog'));
        await user.click(dialog.getByRole('button', { name: /Source export/ }));
        await user.click(await screen.findByRole('option', { name: 'OpenVINO · INT8' }));
        await user.click(dialog.getByRole('button', { name: 'Export RTC' }));
        expect(await dialog.findByRole('status')).toHaveTextContent('Optimization pending');
        await user.click(dialog.getByRole('button', { name: 'Close' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(screen.getByRole('status')).toHaveTextContent('1 variant job in progress');
    });

    it('does not offer creation from an incomplete export', async () => {
        server.use(
            http.get('/api/models/{model_id}', () => HttpResponse.json(getMockedModelDetailResponse({ exports: [] })))
        );
        render(<ModelFormats model={getMockedModel()} />);
        expect(await screen.findByRole('button', { name: 'Add variant' })).toBeDisabled();
        expect(screen.getByText('A complete backend export is required')).toBeInTheDocument();
    });

    it('opens the same creation dialog with the selected card preselected, without a sticky Add variant default', async () => {
        const user = userEvent.setup();
        render(<ModelFormats model={getMockedModel()} />);
        const card = within(await screen.findByRole('region', { name: 'OpenVINO · INT8' }));
        await user.click(card.getByRole('button', { name: 'Actions for OpenVINO INT8' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Optimize' }));
        const dialog = within(await screen.findByRole('dialog', { name: 'Create model variant' }));
        expect(dialog.getByRole('button', { name: /Source export/ })).toHaveTextContent('OpenVINO · INT8');
        expect(dialog.getByRole('button', { name: 'Export RTC' })).toBeInTheDocument();
        await user.click(dialog.getByRole('button', { name: 'Close' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        await user.click(screen.getByRole('button', { name: 'Add variant' }));
        const added = within(await screen.findByRole('dialog', { name: 'Create model variant' }));
        expect(added.getByRole('button', { name: /Source export/ })).toHaveTextContent('OpenVINO · Original');
    });

    it('preselects the backend as well as the variant when opening Optimize from a card', async () => {
        const user = userEvent.setup();
        render(<ModelFormats model={getMockedModel()} />);
        const card = within(await screen.findByRole('region', { name: 'ONNX · Original' }));
        await user.click(card.getByRole('button', { name: 'Actions for ONNX Original' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Optimize' }));
        expect(
            within(await screen.findByRole('dialog')).getByRole('button', { name: /Source export/ })
        ).toHaveTextContent('ONNX · Original');
    });
});
