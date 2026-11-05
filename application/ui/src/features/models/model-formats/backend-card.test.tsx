import { rs } from '@rstest/core';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { getMockedModelDetailResponse } from '../../../test-utils/mocks/mock-model-detail-response';
import { render } from '../../../test-utils/render';
import { BackendCard } from './backend-card';

describe('BackendCard', () => {
    it('keeps variant actions in a compact menu and downloads precisely that package', async () => {
        const user = userEvent.setup();
        const optimize = rs.fn();
        render(
            <BackendCard
                model={getMockedModel()}
                backendType='openvino'
                variant={getMockedExportVariant({
                    id: 'rtc-int8',
                    label: 'RTC + INT8',
                    weight_compression: 'int8_sym',
                })}
                onOptimize={optimize}
            />
        );
        const card = within(screen.getByRole('region', { name: 'OpenVINO · RTC + INT8' }));
        expect(card.getByText('INT8 symmetric weights')).toBeInTheDocument();
        expect(card.getAllByRole('button')).toHaveLength(1);
        await user.click(card.getByRole('button', { name: 'Actions for OpenVINO RTC + INT8' }));
        expect(screen.getByRole('menuitem', { name: 'Download' })).toHaveAttribute(
            'href',
            expect.stringContaining('variant=rtc-int8')
        );
        expect(screen.getByRole('menuitem', { name: 'Runtime export' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
        await user.click(screen.getByRole('menuitem', { name: 'Optimize' }));
        expect(optimize).toHaveBeenCalledTimes(1);
    });

    it('does not offer optimization when no operation is available and never deletes the original', async () => {
        const user = userEvent.setup();
        render(
            <BackendCard
                model={getMockedModel()}
                backendType='openvino'
                variant={getMockedExportVariant({ available_operations: [] })}
                onOptimize={rs.fn()}
            />
        );
        await user.click(screen.getByRole('button', { name: 'Actions for OpenVINO Original' }));
        expect(screen.queryByRole('menuitem', { name: 'Optimize' })).not.toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Delete' })).not.toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Download' })).toHaveAttribute(
            'href',
            expect.stringContaining('variant=original')
        );
        expect(screen.queryByText('FP16')).not.toBeInTheDocument();
    });

    it('only offers runtime recipes for supported backends', async () => {
        const user = userEvent.setup();
        render(
            <BackendCard
                model={getMockedModel()}
                backendType='onnx'
                variant={getMockedExportVariant()}
                onOptimize={rs.fn()}
            />
        );
        await user.click(screen.getByRole('button', { name: 'Actions for ONNX Original' }));
        expect(screen.queryByRole('menuitem', { name: 'Runtime export' })).not.toBeInTheDocument();
    });

    it('opens runtime export with the card variant fixed', async () => {
        const user = userEvent.setup();
        const model = getMockedModel({ dataset_id: null });
        const variant = getMockedExportVariant({ id: 'int8', label: 'INT8' });
        const environment = getMockedEnvironment();
        server.use(
            http.get('/api/models/{model_id}', () =>
                HttpResponse.json(
                    getMockedModelDetailResponse({
                        model,
                        exports: [
                            {
                                type: 'openvino',
                                size_bytes: 1,
                                file_count: 1,
                                variants: [getMockedExportVariant(), variant],
                            },
                        ],
                    })
                )
            ),
            http.get('/api/projects/{project_id}/environments', () =>
                HttpResponse.json([{ id: environment.id, name: environment.name }])
            ),
            http.get('/api/system/devices/inference', () =>
                HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
            )
        );
        render(<BackendCard model={model} backendType='openvino' variant={variant} onOptimize={rs.fn()} />);
        await user.click(screen.getByRole('button', { name: 'Actions for OpenVINO INT8' }));
        await user.click(screen.getByRole('menuitem', { name: 'Runtime export' }));
        const dialog = within(await screen.findByRole('dialog', { name: 'Download runtime export' }));
        expect(dialog.queryByRole('button', { name: /Variant/ })).not.toBeInTheDocument();
        await waitFor(() =>
            expect(dialog.getByRole('button', { name: 'Download' })).toHaveAttribute(
                'href',
                expect.stringContaining('variant=int8')
            )
        );
    });

    it('confirms deletion of the selected variant and keeps an error open for retry', async () => {
        const user = userEvent.setup();
        server.use(
            http.delete('/api/models/{model_id}/exports/{backend}/variants/{variant}', ({ params, response }) => {
                expect(params).toEqual({ model_id: 'model-1', backend: 'openvino', variant: 'int8' });
                return response.untyped(
                    HttpResponse.json({ message: 'Variant is currently running' }, { status: 409 })
                );
            })
        );
        render(
            <BackendCard
                model={getMockedModel()}
                backendType='openvino'
                variant={getMockedExportVariant({ id: 'int8', label: 'INT8' })}
                onOptimize={rs.fn()}
            />
        );
        await user.click(screen.getByRole('button', { name: 'Actions for OpenVINO INT8' }));
        await user.click(screen.getByRole('menuitem', { name: 'Delete' }));
        const dialog = within(await screen.findByRole('dialog', { name: 'Delete int8?' }));
        await user.click(dialog.getByRole('button', { name: 'Delete' }));
        expect(await dialog.findByRole('alert')).toHaveTextContent('Variant is currently running');
        expect(dialog.getByRole('button', { name: 'Delete' })).toBeEnabled();
        await user.click(dialog.getByRole('button', { name: 'Cancel' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });
});
