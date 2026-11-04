import { HttpResponse } from 'msw';

import { getMockedDataset } from '../src/test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../src/test-utils/mocks/mock-environment';
import { getMockedHomeCapability, getMockedHomePose } from '../src/test-utils/mocks/mock-home-position';
import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { getMockedProject } from '../src/test-utils/mocks/mock-project';
import { getMockedRuntimeSession, getMockedSessionEnvironment } from '../src/test-utils/mocks/mock-runtime-session';
import { getMockedSettings } from '../src/test-utils/mocks/mock-settings';
import { expect, http, test } from './fixtures';

const URL = '/projects/project-1/datasets/dataset-1/record';

test.beforeEach(async ({ network, page }) => {
    network.use(
        http.get('/api/settings', () => HttpResponse.json(getMockedSettings())),
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.get('/api/projects/{project_id}', () =>
            HttpResponse.json(getMockedProject({ datasets: [getMockedDataset()] }))
        ),
        http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset())),
        http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([])),
        http.get('/api/projects/{project_id}/environments/{environment_id}', () =>
            HttpResponse.json(getMockedEnvironment())
        ),
        http.get('/api/projects/{project_id}/environments', () =>
            HttpResponse.json([{ id: 'environment-1', name: 'Workbench' }])
        ),
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        ),
        http.get('/api/models/{model_id}', () => HttpResponse.json({ model: getMockedModel(), exports: [] })),
        http.get('/api/projects/{project_id}/robots/{robot_id}', ({ params }) =>
            HttpResponse.json({
                id: String(params.robot_id),
                name: params.robot_id === 'leader-1' ? 'Leader' : 'Follower',
                type: 'demo-robot',
                unavailable: true,
                payload: {},
            })
        )
    );
    // All sockets are intercepted; these tests cannot command physical hardware.
    await page.routeWebSocket('**/api/**', (socket) => {
        if (!socket.url().includes('/runtime/ws')) return;
        const state = getMockedRuntimeSession({
            environment: getMockedSessionEnvironment({
                leader_id: 'leader-1',
                home_capability: getMockedHomeCapability(),
            }),
            model: { model_id: 'model-1', name: 'Cube policy', backend: 'onnx', device: 'cpu' },
        });
        const publish = () => {
            socket.send(JSON.stringify({ type: 'state', data: state }));
            for (const component of ['environment/follower', 'environment/leader'])
                socket.send(JSON.stringify({ type: 'status', component, state: 'running' }));
        };
        socket.onMessage((message) => {
            const command = JSON.parse(String(message));
            let result;
            if (command.command === 'set_control') state.control = command.control;
            if (command.command === 'capture_home_position') result = getMockedHomePose();
            if (command.command === 'return_home') {
                state.control = null;
                state.home_motion = {
                    operation_id: command.operation_id,
                    robot_id: 'robot-1',
                    state: 'moving',
                    progress: 0.2,
                    sequence: 1,
                };
                result = state.home_motion;
            }
            if (command.command === 'renew_home_return') result = state.home_motion;
            if (
                command.command === 'cancel_home_return' &&
                state.home_motion != null &&
                state.home_motion.operation_id === command.operation_id
            ) {
                state.home_motion = {
                    ...state.home_motion,
                    state: 'cancelled',
                    sequence: 2,
                    message: 'Return stopped.',
                };
                result = state.home_motion;
            }
            publish();
            socket.send(JSON.stringify({ type: 'ack', request_id: command.request_id, ok: true, result }));
        });
        publish();
    });
});

test('captures home, confirms return during inference, switches mode, and keeps Stop visible', async ({
    page,
}, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.goto(URL);
    await page.getByRole('button', { name: 'Set current pose as home' }).click();
    const home = page.getByRole('region', { name: 'Home position', exact: true });
    await expect(home.getByRole('button', { name: 'Return to home' })).toBeEnabled();
    await page.getByRole('radio', { name: 'Inference', exact: true }).click();
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Stop model' })).toBeVisible();
    await home.getByRole('button', { name: 'Return to home' }).click();
    const dialog = page.getByRole('dialog');
    await expect(dialog.getByRole('button', { name: 'Return to home' })).toBeDisabled();
    await dialog.getByRole('checkbox', { name: /return path is clear/ }).check();
    await page.screenshot({ path: testInfo.outputPath('home-confirmation.png') });
    await dialog.getByRole('button', { name: 'Return to home' }).click();
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeChecked();
    const stop = page.getByRole('button', { name: 'Stop return' });
    await expect(stop).toBeEnabled();
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeDisabled();
    await expect(page.getByRole('switch', { name: 'Teleoperate' })).toBeDisabled();
    await page.getByRole('complementary', { name: 'Session setup' }).evaluate((element) => {
        element.scrollTop = element.scrollHeight;
    });
    await expect(stop).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('home-return.png') });
    await stop.click();
    await expect(stop).toBeHidden();
    await expect(page.getByRole('button', { name: 'Stop model' })).toBeHidden();
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
});

test('teleoperation is a direct takeover from inference and selects Recording', async ({ page }, testInfo) => {
    await page.goto(URL);
    await page.getByRole('radio', { name: 'Inference', exact: true }).click();
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Stop model' })).toBeVisible();
    await page.getByRole('switch', { name: 'Teleoperate' }).click();
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeChecked();
    await expect(page.getByRole('switch', { name: 'Teleoperate' })).toBeChecked();
    await expect(page.getByRole('button', { name: 'Stop model' })).toBeHidden();
    await page.screenshot({ path: testInfo.outputPath('teleop-takeover.png') });
});

test('restores browser home without automatically moving on reload', async ({ page }) => {
    await page.goto(URL);
    await page.getByRole('button', { name: 'Set current pose as home' }).click();
    await expect(page.getByRole('button', { name: 'Return to home', exact: true })).toBeEnabled();
    await page.reload();
    await expect(page.getByRole('button', { name: 'Return to home', exact: true })).toBeEnabled();
    await expect(page.getByRole('region', { name: 'Home return controls' })).toBeHidden();
});
