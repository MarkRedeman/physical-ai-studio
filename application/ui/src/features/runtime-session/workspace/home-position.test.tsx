import { rs } from '@rstest/core';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { getMockedHomeCapability, getMockedHomePose } from '../../../test-utils/mocks/mock-home-position';
import { getMockedRuntimeSession, getMockedSessionEnvironment } from '../../../test-utils/mocks/mock-runtime-session';
import { render } from '../../../test-utils/render';
import { HomePosition, HomeReturnBanner } from './home-position';
import { HomePositionProvider } from './home-position-provider';
import { homeStorageKey } from './home-position-storage';
import { SessionModeProvider, useSessionMode } from './session-mode';

const send = rs.fn();
let state = getMockedRuntimeSession();
rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: { timeout: 1000 },
    failureOf: () => null,
    useRuntimeSession: () => ({
        state,
        send,
        statuses: { 'environment/follower': { state: 'running' } },
        isLoading: false,
    }),
}));
rs.mock('./part-controls', () => ({ usePartControls: () => ({ pendingPart: null }) }));
const Mode = () => <output aria-label='Selected mode'>{useSessionMode().mode}</output>;
const renderHome = () =>
    render(
        <HomePositionProvider>
            <SessionModeProvider initialMode='inference'>
                <HomePosition />
                <HomeReturnBanner />
                <Mode />
            </SessionModeProvider>
        </HomePositionProvider>,
        { route: '/projects/project-1', path: '/projects/:project_id' }
    );

beforeEach(() => {
    window.localStorage.clear();
    state = getMockedRuntimeSession({
        environment: getMockedSessionEnvironment({ home_capability: getMockedHomeCapability() }),
    });
    send.mockReset().mockImplementation(async (command) => {
        const result =
            command.command === 'capture_home_position'
                ? getMockedHomePose()
                : {
                      operation_id: command.operation_id,
                      robot_id: 'robot-1',
                      progress: 0.2,
                      state: command.command === 'cancel_home_return' ? 'cancelled' : 'moving',
                      sequence: command.command === 'cancel_home_return' ? 2 : 1,
                      message: command.command === 'cancel_home_return' ? 'Return stopped.' : null,
                  };
        return { ok: true, result };
    });
});

const restorePose = () =>
    window.localStorage.setItem(
        homeStorageKey('project-1', getMockedSessionEnvironment()) ?? '',
        JSON.stringify(getMockedHomePose())
    );

it('restores a saved home without sending any motion command', async () => {
    restorePose();
    renderHome();
    expect(await screen.findByRole('button', { name: 'Return to home' })).toBeEnabled();
    expect(send).not.toHaveBeenCalled();
});

it('captures backend measurements and persists the validated pose', async () => {
    const user = userEvent.setup();
    renderHome();
    await user.click(screen.getByRole('button', { name: 'Set current pose as home' }));
    expect(await screen.findByRole('button', { name: 'Return to home' })).toBeEnabled();
    expect(send).toHaveBeenCalledWith({ command: 'capture_home_position', robot_id: 'robot-1' });
    expect(window.localStorage.getItem(homeStorageKey('project-1', getMockedSessionEnvironment()) ?? '')).toContain(
        'shoulder_pan.pos'
    );
});

it('requires path confirmation, switches to Recording after takeover, and keeps Stop independent', async () => {
    restorePose();
    const user = userEvent.setup();
    renderHome();
    await user.click(await screen.findByRole('button', { name: 'Return to home' }));
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByRole('button', { name: 'Return to home' })).toBeDisabled();
    await user.click(within(dialog).getByRole('checkbox', { name: /return path is clear/ }));
    await user.click(within(dialog).getByRole('button', { name: 'Return to home' }));
    expect(await screen.findByRole('button', { name: 'Stop return' })).toBeEnabled();
    await waitFor(() => expect(screen.getByLabelText('Selected mode')).toHaveTextContent('record'));
    await user.click(screen.getByRole('button', { name: 'Stop return' }));
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Stop return' })).not.toBeInTheDocument());
    expect(screen.getByText('Return stopped.')).toBeInTheDocument();
});

it('does not use a stale home and offers setting a new one', async () => {
    window.localStorage.setItem(
        homeStorageKey('project-1', getMockedSessionEnvironment()) ?? '',
        JSON.stringify(getMockedHomePose({ fingerprint: 'b'.repeat(64) }))
    );
    renderHome();
    expect(await screen.findByText(/Saved home is invalid/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Return to home' })).not.toBeInTheDocument();
});
