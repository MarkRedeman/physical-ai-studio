import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useLocation } from 'react-router';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { render } from '../../../test-utils/render';
import { RunVariant } from './run-variant';

const Location = () => {
    const { pathname, search } = useLocation();
    return <output aria-label='Current location'>{pathname + search}</output>;
};

describe('RunVariant', () => {
    it.each(['original', 'rtc-int8', 'snapflow-rtc-int8'])(
        'opens inference directly for %s using the preferred compatible device',
        async (variant) => {
            server.use(
                http.get('/api/system/devices/inference', () =>
                    HttpResponse.json([
                        { type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' },
                        { type: 'npu', name: 'NPU', backend: 'openvino', device: 'NPU' },
                        { type: 'xpu', name: 'NVIDIA GPU', backend: 'openvino', device: 'GPU.0' },
                        { type: 'xpu', name: 'Intel GPU', backend: 'openvino', device: 'GPU.1' },
                    ])
                )
            );
            const user = userEvent.setup();
            render(
                <>
                    <RunVariant
                        model={getMockedModel()}
                        backend='openvino'
                        variant={getMockedExportVariant({ id: variant })}
                    />
                    <Location />
                </>,
                { path: '*', route: '/projects/project-1/models' }
            );
            const button = screen.getByRole('button', { name: 'Run' });
            expect(button).toBeDisabled();
            await waitFor(() => expect(button).toBeEnabled());
            await user.click(button);
            expect(screen.getByLabelText('Current location')).toHaveTextContent(
                `/projects/project-1/models/model-1/inference/openvino?device=GPU.1&variant=${variant}`
            );
            expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        }
    );

    it('uses the card backend rather than the model default and falls back to CPU', async () => {
        server.use(
            http.get('/api/system/devices/inference', () =>
                HttpResponse.json([
                    { type: 'xpu', name: 'Intel GPU', backend: 'openvino', device: 'GPU' },
                    { type: 'cpu', name: 'CPU', backend: 'torch', device: 'cpu' },
                ])
            )
        );
        const user = userEvent.setup();
        render(
            <>
                <RunVariant model={getMockedModel()} backend='torch' variant={getMockedExportVariant({ id: 'int8' })} />
                <Location />
            </>,
            { path: '*', route: '/' }
        );
        await waitFor(() => expect(screen.getByRole('button', { name: 'Run' })).toBeEnabled());
        await user.click(screen.getByRole('button', { name: 'Run' }));
        expect(screen.getByLabelText('Current location')).toHaveTextContent('/inference/torch?device=cpu&variant=int8');
    });

    it('does not query devices or run unsupported exports', async () => {
        const devices = rs.fn();
        server.use(
            http.get('/api/system/devices/inference', () => {
                devices();
                return HttpResponse.json([]);
            })
        );
        const user = userEvent.setup();
        render(
            <RunVariant
                model={getMockedModel()}
                backend='executorch'
                variant={getMockedExportVariant({ can_run_in_studio: false })}
            />
        );
        expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled();
        await user.click(screen.getByRole('button', { name: 'Why Run is unavailable' }));
        expect(await screen.findByText('This export cannot run in Studio.')).toBeInTheDocument();
        expect(devices).not.toHaveBeenCalled();
        await user.click(screen.getByRole('button', { name: 'Dismiss' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });

    it('does not silently switch backends when no compatible device is available', async () => {
        server.use(
            http.get('/api/system/devices/inference', () =>
                HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'torch', device: 'cpu' }])
            )
        );
        const user = userEvent.setup();
        render(<RunVariant model={getMockedModel()} backend='openvino' variant={getMockedExportVariant()} />);
        const button = screen.getByRole('button', { name: 'Run' });
        await user.click(await screen.findByRole('button', { name: 'Why Run is unavailable' }));
        expect(button).toBeDisabled();
        expect(
            await screen.findByText('No compatible inference device is available for this backend.')
        ).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Dismiss' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });

    it('can retry a failed device query before running', async () => {
        server.use(http.get('/api/system/devices/inference', () => HttpResponse.error()));
        const user = userEvent.setup();
        render(<RunVariant model={getMockedModel()} backend='openvino' variant={getMockedExportVariant()} />);
        const retry = await screen.findByRole('button', { name: 'Retry devices' });
        expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled();
        server.use(
            http.get('/api/system/devices/inference', () =>
                HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
            )
        );
        await user.click(retry);
        await waitFor(() => expect(screen.getByRole('button', { name: 'Run' })).toBeEnabled());
    });
});
