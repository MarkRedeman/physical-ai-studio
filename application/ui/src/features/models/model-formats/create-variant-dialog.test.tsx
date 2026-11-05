import { rs } from '@rstest/core';
import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { getMockedModelDetailResponse } from '../../../test-utils/mocks/mock-model-detail-response';
import { render } from '../../../test-utils/render';
import { ModelFormats } from './model-formats';

const { loading } = rs.hoisted(() => ({ loading: { ready: false, promise: Promise.resolve() } }));

rs.mock('./optimize-model', () => ({
    OptimizeModel: () => {
        if (!loading.ready) throw loading.promise;
        return <span>Optimization controls loaded</span>;
    },
}));

it('keeps the model row visible while optimization dialog content suspends', async () => {
    let release = () => {};
    loading.ready = false;
    loading.promise = new Promise<void>((resolve) => {
        release = resolve;
    });
    server.use(
        http.get('/api/models/{model_id}', () => HttpResponse.json(getMockedModelDetailResponse())),
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.get('/api/system/devices/inference', () =>
            HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
        )
    );
    const user = userEvent.setup();
    render(<ModelFormats model={getMockedModel()} />);
    const card = await screen.findByRole('region', { name: 'OpenVINO · Original' });
    await user.click(screen.getByRole('button', { name: 'Add variant' }));
    const dialog = within(await screen.findByRole('dialog', { name: 'Create model variant' }));
    expect(dialog.getByRole('progressbar', { name: 'Loading variant options' })).toBeVisible();
    expect(dialog.getByRole('button', { name: 'Close' })).toBeEnabled();
    expect(card).toBeVisible();

    await act(async () => {
        loading.ready = true;
        release();
        await loading.promise;
    });
    expect(await dialog.findByText('Optimization controls loaded')).toBeVisible();
    expect(dialog.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(card).toBeVisible();
    await user.click(dialog.getByRole('button', { name: 'Close' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
});
