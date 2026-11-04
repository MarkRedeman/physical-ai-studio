import { ReactNode } from 'react';

import { rs } from '@rstest/core';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedDataset } from '../../../test-utils/mocks/mock-dataset';
import { getMockedEpisodes } from '../../../test-utils/mocks/mock-episode';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { getMockedRuntimeSession, getMockedSessionDataset } from '../../../test-utils/mocks/mock-runtime-session';
import { getMockedSettings } from '../../../test-utils/mocks/mock-settings';
import { render } from '../../../test-utils/render';
import { Statuses } from '../runtime-session-provider';
import { SessionWorkspace } from './session-workspace';

const send = rs.fn();
const act = rs.fn();
let state = getMockedRuntimeSession();
let statuses: Statuses = {};
let pendingPart: string | null = null;

rs.mock('../runtime-session-provider', () => ({
    useRuntimeSession: () => ({ state, statuses, send, isLoading: false }),
    failureOf: (values: Statuses, part: string) => (values[part]?.state === 'failed' ? values[part].error : null),
}));
rs.mock('../session-viewer', () => ({ SessionViewer: () => <div>Live camera views</div> }));
rs.mock('./part-controls', () => ({
    PartControlsProvider: ({ children }: { children: ReactNode }) => children,
    usePartControls: () => ({ pendingPart, act }),
}));

const renderWorkspace = () =>
    render(<SessionWorkspace initialMode='record' cancelHref='/projects' />, {
        route: '/projects/project-1/datasets/dataset-1/record',
        path: '/projects/:project_id/datasets/:dataset_id/record',
    });

beforeEach(() => {
    state = getMockedRuntimeSession();
    statuses = {};
    pendingPart = null;
    send.mockReset().mockResolvedValue({ ok: true });
    act.mockReset();
    server.use(
        http.get('/api/settings', () => HttpResponse.json(getMockedSettings())),
        http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([])),
        http.get('/api/dataset/{dataset_id}', () =>
            HttpResponse.json(getMockedDataset({ default_task: 'Pick up the cube' }))
        ),
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([]))
    );
});

describe('SessionWorkspace', () => {
    it('keeps the setup panels without an introductory heading or description', async () => {
        renderWorkspace();
        const setup = await screen.findByRole('complementary', { name: 'Session setup' });
        expect(within(setup).queryByRole('heading', { name: 'Session setup' })).not.toBeInTheDocument();
        expect(
            within(setup).queryByText('Select a mode to record demonstrations or run a model.')
        ).not.toBeInTheDocument();
        for (const name of ['Environment', 'Recording', 'Inference']) {
            expect(within(setup).getByRole('region', { name })).toBeInTheDocument();
        }
    });

    it('opens device failures automatically and lets the header chevron hide them', async () => {
        statuses['environment/leader'] = {
            type: 'status',
            component: 'environment/leader',
            state: 'failed',
            error: 'Leader disconnected',
        };
        const user = userEvent.setup();
        renderWorkspace();
        const toggle = await screen.findByRole('button', { name: 'Hide device details' });
        expect(toggle).toHaveAttribute('aria-expanded', 'true');
        expect(screen.getByText('Leader disconnected')).toBeVisible();
        await user.click(toggle);
        expect(screen.getByRole('button', { name: 'Show device details' })).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByText('Leader disconnected')).not.toBeInTheDocument();
    });

    it('shows the dataset total in the recording panel, including this session', async () => {
        state.dataset = getMockedSessionDataset({ episodes_recorded: 3 });
        server.use(http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json(getMockedEpisodes())));
        renderWorkspace();
        const recording = await screen.findByRole('region', { name: 'Recording' });
        expect(await within(recording).findByRole('group', { name: '5 episodes in dataset' })).toBeInTheDocument();
        expect(within(recording).getByRole('button', { name: 'Recording tips' })).toBeInTheDocument();
    });

    it('highlights a failed dataset and opens its error details', async () => {
        statuses.dataset = {
            type: 'status',
            component: 'dataset',
            state: 'failed',
            error: 'Could not open the dataset',
        };
        renderWorkspace();
        const recording = await screen.findByRole('region', { name: 'Recording' });
        expect(recording).toHaveAttribute('data-error', 'true');
        expect(within(recording).getByRole('button', { name: 'Error details' })).toHaveAttribute(
            'aria-expanded',
            'true'
        );
        expect(within(recording).getByText('Could not open the dataset')).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Environment' })).not.toHaveAttribute('data-error');
    });

    it('shows a quiet inference placeholder when the project has no models', async () => {
        renderWorkspace();
        expect(await screen.findByText(/Once you train a model/)).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Choose model/ })).not.toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Recording' })).toBeInTheDocument();
    });

    it('offers loading when the project has an exported model', async () => {
        server.use(
            http.get('/api/projects/{project_id}/models', () =>
                HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
            )
        );
        renderWorkspace();
        const choose = await screen.findByRole('button', { name: 'Choose model…' });
        await waitFor(() => expect(choose).toBeEnabled());
        expect(screen.getByText('No model selected')).toBeInTheDocument();
        expect(screen.queryByText(/Once you train a model/)).not.toBeInTheDocument();
        await userEvent.setup().click(choose);
        expect(act).toHaveBeenCalledExactlyOnceWith('model', 'swap');
        expect(screen.getByRole('radio', { name: 'Recording' })).toBeChecked();
    });

    it('keeps the model prerequisite outside its disabled header action', async () => {
        state.environment = null;
        server.use(
            http.get('/api/projects/{project_id}/models', () =>
                HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
            )
        );
        renderWorkspace();
        const inference = await screen.findByRole('region', { name: 'Inference' });
        const choose = await within(inference).findByRole('button', { name: 'Choose model…' });
        expect(choose).toBeDisabled();
        const guidance = within(inference).getByText('Connect an environment first');
        expect(guidance).toBeVisible();
        expect(choose).not.toContainElement(guidance);
        expect(within(inference).getByText('No model selected')).toHaveAttribute('data-empty', 'true');
    });

    it('shows preparation instead of a second model selector while loading', async () => {
        pendingPart = 'model';
        renderWorkspace();
        const inference = await screen.findByRole('region', { name: 'Inference' });
        expect(within(inference).getByRole('progressbar', { name: 'Preparing model' })).toBeInTheDocument();
        expect(within(inference).getByText('Preparing…')).toBeVisible();
        expect(within(inference).queryByRole('button', { name: 'Choose model…' })).not.toBeInTheDocument();
    });

    it('explains that an unexported model must be exported first', async () => {
        server.use(http.get('/api/projects/{project_id}/models', () => HttpResponse.json([getMockedModel()])));
        renderWorkspace();
        expect(await screen.findByText('Export a trained model to run it here.')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Choose model/ })).not.toBeInTheDocument();
    });

    it('starts with the task edited in the recording card', async () => {
        const user = userEvent.setup();
        renderWorkspace();
        const task = await screen.findByRole('textbox', { name: /Current task/ });
        await waitFor(() => expect(task).toHaveValue('Pick up the cube'));
        await user.clear(task);
        await user.type(task, 'Place the cube in the tray');
        await user.click(screen.getByRole('button', { name: /Start episode/ }));
        expect(send).toHaveBeenCalledExactlyOnceWith({ command: 'start_episode', task: 'Place the cube in the tray' });
    });

    it('registers just one start hotkey and ignores it while typing', async () => {
        renderWorkspace();
        const task = await screen.findByRole('textbox', { name: /Current task/ });
        await waitFor(() => expect(task).toHaveValue('Pick up the cube'));
        fireEvent.keyDown(task, { key: 'ArrowRight' });
        expect(send).not.toHaveBeenCalled();
        fireEvent.keyDown(window, { key: 'ArrowRight' });
        await waitFor(() =>
            expect(send).toHaveBeenCalledExactlyOnceWith({ command: 'start_episode', task: 'Pick up the cube' })
        );
    });

    it('does not start an episode with a whitespace-only task', async () => {
        const user = userEvent.setup();
        renderWorkspace();
        const task = await screen.findByRole('textbox', { name: /Current task/ });
        await waitFor(() => expect(task).toHaveValue('Pick up the cube'));
        await user.clear(task);
        await user.type(task, '   ');
        expect(screen.getByRole('button', { name: /Start episode/ })).toBeDisabled();
        fireEvent.keyDown(window, { key: 'ArrowRight' });
        expect(send).not.toHaveBeenCalled();
    });

    it('blocks starting when the environment has failed', async () => {
        statuses.environment = { type: 'status', component: 'environment', state: 'failed', error: 'Disconnected' };
        renderWorkspace();
        expect(await screen.findByRole('button', { name: /Start episode/ })).toBeDisabled();
        expect(screen.getByText('Load a working environment to record episodes')).toBeInTheDocument();
    });

    it('blocks starting while a resource is being replaced', async () => {
        pendingPart = 'model';
        renderWorkspace();
        const task = await screen.findByRole('textbox', { name: /Current task/ });
        await waitFor(() => expect(task).toHaveValue('Pick up the cube'));
        expect(screen.getByRole('button', { name: /Start episode/ })).toBeDisabled();
    });

    it('keeps save and discard available after a device fails during recording', async () => {
        state = getMockedRuntimeSession({ dataset: getMockedSessionDataset({ is_recording: true }) });
        statuses.environment = { type: 'status', component: 'environment', state: 'failed', error: 'Disconnected' };
        const user = userEvent.setup();
        renderWorkspace();
        expect(await screen.findByRole('button', { name: /Save episode/ })).toBeEnabled();
        expect(screen.getByRole('button', { name: /Discard/ })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'dataset actions' })).toBeDisabled();
        expect(screen.getByRole('textbox', { name: /Current task/ })).toHaveAttribute('readonly');
        await user.click(screen.getByRole('button', { name: /Save episode/ }));
        expect(send).toHaveBeenCalledExactlyOnceWith({ command: 'save_episode' });
    });

    it('shows loaded model controls even if the model list is empty', async () => {
        state.model = { model_id: 'model-1', name: 'Cube policy', backend: 'onnx', device: 'cpu' };
        server.use(
            http.get('/api/models/{model_id}', () =>
                HttpResponse.json({ model: getMockedModel({ dataset_id: null }), exports: [] })
            )
        );
        const user = userEvent.setup();
        renderWorkspace();
        const inference = await screen.findByRole('region', { name: 'Inference' });
        await user.click(within(inference).getByRole('radio', { name: 'Inference' }));
        expect(screen.getByRole('button', { name: 'Run model' })).toBeEnabled();
        expect(within(inference).queryByRole('slider')).not.toBeInTheDocument();
        await user.click(within(inference).getByRole('button', { name: /Execution settings/ }));
        expect(within(inference).getAllByRole('slider')).toHaveLength(2);
        await user.click(within(inference).getByRole('radio', { name: 'Asynchronous' }));
        expect(within(inference).getAllByRole('slider')).toHaveLength(2);
    });

    it('prevents starting model control during an episode', async () => {
        state = getMockedRuntimeSession({
            dataset: getMockedSessionDataset({ is_recording: true }),
            model: { model_id: 'model-1', name: 'Cube policy', backend: 'onnx', device: 'cpu' },
        });
        renderWorkspace();
        expect(await screen.findByRole('radio', { name: 'Inference' })).toBeDisabled();
        expect(screen.queryByText(/before switching modes/)).not.toBeInTheDocument();
    });
});
