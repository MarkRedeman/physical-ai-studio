import { rs } from '@rstest/core';
import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useLocation } from 'react-router';

import { getMockedRuntimeSession, getMockedSessionDataset } from '../../../test-utils/mocks/mock-runtime-session';
import { render } from '../../../test-utils/render';
import { BackToDatasetButton } from './back-to-dataset-button';

const send = rs.fn();
let state = getMockedRuntimeSession();
rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: { timeout: 10_000 },
    useRuntimeSession: () => ({ state, send }),
}));

const CurrentPath = () => <output aria-label='Current path'>{useLocation().pathname}</output>;
const renderButton = () =>
    render(
        <>
            <BackToDatasetButton dataset_id='dataset-1' />
            <CurrentPath />
        </>,
        {
            route: '/projects/project-1/datasets/dataset-1/record',
            path: '/projects/:project_id/*',
        }
    );

beforeEach(() => {
    state = getMockedRuntimeSession();
    send.mockReset().mockResolvedValue({ ok: true });
});

it('waits for the dataset to finish saving before leaving', async () => {
    let finish = () => {};
    send.mockImplementation(
        () =>
            new Promise<{ ok: boolean }>((resolve) => {
                finish = () => resolve({ ok: true });
            })
    );
    const user = userEvent.setup();
    renderButton();
    await user.click(screen.getByRole('button', { name: 'Back to dataset' }));
    expect(send).toHaveBeenCalledExactlyOnceWith({ command: 'unload_dataset' });
    expect(screen.getByLabelText('Current path')).toHaveTextContent('/record');
    expect(screen.getByRole('button', { name: 'Back to dataset' })).toBeDisabled();
    await act(async () => finish());
    await waitFor(() =>
        expect(screen.getByLabelText('Current path')).toHaveTextContent(/^\/projects\/project-1\/datasets\/dataset-1$/)
    );
});

it('stays in the session if saving fails', async () => {
    send.mockResolvedValue({ ok: false, error: { message: 'Disk full' } });
    const user = userEvent.setup();
    renderButton();
    await user.click(screen.getByRole('button', { name: 'Back to dataset' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Back to dataset' })).toBeEnabled());
    expect(screen.getByLabelText('Current path')).toHaveTextContent('/record');
});

it('requires saving or discarding an active episode before leaving', () => {
    state.dataset = getMockedSessionDataset({ is_recording: true });
    renderButton();
    expect(screen.getByRole('button', { name: 'Back to dataset' })).toBeDisabled();
    expect(send).not.toHaveBeenCalled();
});

it('returns to the selected dataset after it was changed', async () => {
    state.dataset = getMockedSessionDataset({ dataset_id: 'dataset-2' });
    const user = userEvent.setup();
    renderButton();
    await user.click(screen.getByRole('button', { name: 'Back to dataset' }));
    await waitFor(() =>
        expect(screen.getByLabelText('Current path')).toHaveTextContent(/^\/projects\/project-1\/datasets\/dataset-2$/)
    );
});
