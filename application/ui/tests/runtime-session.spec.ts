import { HttpResponse } from 'msw';

import { getMockedDataset } from '../src/test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../src/test-utils/mocks/mock-environment';
import { getMockedEpisodes } from '../src/test-utils/mocks/mock-episode';
import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { getMockedProject } from '../src/test-utils/mocks/mock-project';
import { getMockedRuntimeSession, getMockedSessionCamera } from '../src/test-utils/mocks/mock-runtime-session';
import { getMockedSettings } from '../src/test-utils/mocks/mock-settings';
import { expect, http, test } from './fixtures';

const RECORD_URL = '/projects/project-1/datasets/dataset-1/record';

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
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([])),
        http.get('/api/projects/{project_id}/robots/{robot_id}', ({ params }) =>
            HttpResponse.json({
                id: String(params.robot_id),
                name: params.robot_id === 'leader-1' ? 'Demo leader' : 'Demo follower',
                type: 'demo-robot',
                unavailable: true,
                payload: {},
            })
        )
    );

    // Intercept every socket: these tests must never acquire hardware or change a real session.
    await page.routeWebSocket('**/api/**', (socket) => {
        if (!socket.url().includes('/runtime/ws')) return;
        const state = getMockedRuntimeSession();
        if (state.environment) state.environment.leader_id = 'leader-1';
        if (state.environment && page.url().includes('devicesPreview'))
            state.environment.cameras = [getMockedSessionCamera()];
        if (state.dataset && page.url().includes('newDataset')) state.dataset.fps_fixed = false;
        if (state.dataset && page.url().includes('summaryPreview')) state.dataset.episodes_recorded = 6;
        if (page.url().includes('modePreview'))
            state.model = {
                model_id: 'model-1',
                name: 'Cube policy',
                backend: 'onnx',
                device: 'cpu',
            };
        const publish = () => {
            socket.send(JSON.stringify({ type: 'state', data: state }));
            for (const component of ['environment/follower', 'environment/leader']) {
                const failed = page.url().includes('deviceFailure') && component === 'environment/leader';
                socket.send(
                    JSON.stringify({
                        type: 'status',
                        component,
                        state: failed ? 'failed' : 'running',
                        error: failed ? 'Leader disconnected. Check its USB connection.' : null,
                    })
                );
            }
            if (state.environment?.cameras.length)
                socket.send(JSON.stringify({ type: 'status', component: 'environment/wrist', state: 'running' }));
        };
        socket.onMessage((message) => {
            const command = JSON.parse(String(message));
            if (command.command === 'set_control') state.control = command.control;
            if (command.command === 'load_dataset' && state.dataset && command.fps) state.dataset.fps = command.fps;
            if (command.command === 'start_episode' && state.dataset) state.dataset.is_recording = true;
            if (command.command === 'save_episode' && state.dataset) {
                state.dataset.is_recording = false;
                state.dataset.episodes_recorded += 1;
            }
            if (command.command === 'discard_episode' && state.dataset) state.dataset.is_recording = false;
            publish();
            socket.send(JSON.stringify({ type: 'ack', request_id: command.request_id, ok: true }));
        });
        publish();
    });
});

test('keeps inference secondary before the first model is trained', async ({ page }, testInfo) => {
    await page.goto(RECORD_URL);
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
    await expect(page.getByText(/Once you train a model/)).toBeVisible();
    await expect(page.getByRole('button', { name: /Choose model/ })).toBeHidden();
    await expect(page.getByRole('region', { name: 'Recording' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Back to dataset' })).toBeVisible();
    await expect(page.getByRole('group', { name: '0 episodes in dataset' })).toBeVisible();
    await expect(page.getByText('Plugin unavailable:')).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('ready.png') });
});

test('keeps episode actions visible and locks resource changes while recording', async ({ page }, testInfo) => {
    await page.goto(RECORD_URL);
    await page.getByRole('switch', { name: 'Teleoperate' }).click();
    await expect(page.getByRole('switch', { name: 'Teleoperate' })).toBeChecked();
    await page.getByRole('button', { name: /Start episode/ }).click();
    await expect(page.getByRole('button', { name: /Save episode/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'dataset actions' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Back to dataset' })).toBeDisabled();
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeDisabled();
    await expect(page.getByRole('radio', { name: 'Inference', exact: true })).toBeDisabled();
    await expect(page.getByRole('switch', { name: 'Teleoperate' })).toBeDisabled();
    await expect(page.getByRole('textbox', { name: /Current task/ })).toHaveAttribute('readonly');
    await page.screenshot({ path: testInfo.outputPath('recording.png') });
    await page.getByRole('button', { name: /Save episode/ }).click();
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
    await expect(page.getByText('Plugin unavailable:')).toBeVisible();
    await expect(page.getByText('+1 this session')).toBeVisible();
});

test('previews mode radios and moves the glow without starting robot motion', async ({ page, network }, testInfo) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        ),
        http.get('/api/models/{model_id}', () => HttpResponse.json({ model: getMockedModel(), exports: [] }))
    );
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.goto(`${RECORD_URL}?modePreview=true`);
    const recording = page.getByRole('region', { name: 'Recording', exact: true });
    const inference = page.getByRole('region', { name: 'Inference', exact: true });
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeChecked();
    await expect(recording).toHaveAttribute('data-active', 'true');
    await expect(inference).not.toHaveAttribute('data-active');
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
    await page.screenshot({ path: testInfo.outputPath('recording-selected.png') });
    await page.getByRole('radio', { name: 'Inference', exact: true }).click();
    await expect(inference).toHaveAttribute('data-active', 'true');
    await expect(recording).not.toHaveAttribute('data-active');
    await expect(page.getByRole('button', { name: 'Run model', exact: true })).toBeEnabled();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeHidden();
    await expect(recording).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('inference-selected.png') });
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeVisible();
    await page.getByRole('radio', { name: 'Recording', exact: true }).click();
    await expect(recording).toHaveAttribute('data-active', 'true');
    await expect(page.getByRole('switch', { name: 'Teleoperate' })).not.toBeChecked();
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
    await page.getByRole('radio', { name: 'Recording', exact: true }).focus();
    await page.keyboard.press('ArrowDown');
    await expect(page.getByRole('radio', { name: 'Inference', exact: true })).toBeChecked();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeHidden();
});

test('stacks setup below the live views on narrower screens', async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 820, height: 1000 });
    await page.goto(RECORD_URL);
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
    const live = await page.getByRole('main', { name: 'Live session' }).boundingBox();
    const setup = await page.getByRole('complementary', { name: 'Session setup' }).boundingBox();
    expect(live).not.toBeNull();
    expect(setup).not.toBeNull();
    expect(setup?.y).toBeGreaterThanOrEqual((live?.y ?? 0) + (live?.height ?? 0));
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('narrow.png') });
});

test('highlights a disconnected device and opens its details automatically', async ({ page }, testInfo) => {
    await page.goto(`${RECORD_URL}?deviceFailure=true`);
    const environment = page.getByRole('region', { name: 'Environment' });
    await expect(environment).toHaveAttribute('data-error', 'true');
    await expect(environment.getByRole('button', { name: 'Hide device details' })).toHaveAttribute(
        'aria-expanded',
        'true'
    );
    await expect(environment.getByText('Leader disconnected. Check its USB connection.')).toBeVisible();
    await expect(environment.getByRole('switch', { name: 'Teleoperate' })).toBeDisabled();
    await page.screenshot({ path: testInfo.outputPath('device-error.png') });
});

test('shows robot representations and activity inside the environment chevron', async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 1050 });
    await page.goto(`${RECORD_URL}?devicesPreview=true`);
    const environment = page.getByRole('region', { name: 'Environment' });
    await environment.getByRole('button', { name: 'Show device details' }).click();
    await expect(environment.getByText('Demo follower')).toBeVisible();
    await expect(environment.getByText('Demo leader')).toBeVisible();
    await expect(environment.getByRole('group', { name: 'Camera: Wrist camera' }).getByText('Streaming')).toBeVisible();
    await page.getByRole('switch', { name: 'Teleoperate' }).click();
    await expect(environment.getByRole('group', { name: 'Follower robot' }).getByText('Active')).toBeVisible();
    await expect(environment.getByRole('group', { name: 'Leader robot' }).getByText('Active')).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('robot-overview.png') });
});

test('previews a searchable task history', async ({ page, network }, testInfo) => {
    network.use(
        http.get('/api/dataset/{dataset_id}/episodes', () =>
            HttpResponse.json(
                getMockedEpisodes([
                    { episode_index: 0, tasks: ['Pick up the cube and place it in the tray.'] },
                    {
                        episode_index: 1,
                        tasks: [
                            'Move the red block to the left side of the table, then return the arm to its starting position.',
                        ],
                    },
                ])
            )
        )
    );
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(RECORD_URL);
    await page.getByRole('button', { name: 'Use previous task…' }).click();
    await page.getByRole('searchbox', { name: 'Find a task' }).fill('red block');
    await page.screenshot({ path: testInfo.outputPath('task-history.png') });
    await page.getByRole('button', { name: /Move the red block/ }).click();
    await expect(page.getByRole('textbox', { name: /Current task/ })).toHaveValue(/Move the red block/);
    await page.screenshot({ path: testInfo.outputPath('recording-setup.png') });
});

test('previews recording rate before the first episode', async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(`${RECORD_URL}?newDataset=true`);
    await expect(page.getByRole('slider', { name: 'Recording rate' })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('rate-first.png') });
});

test('previews the actionable inference well', async ({ page, network }, testInfo) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        )
    );
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(RECORD_URL);
    const well = page.getByRole('button', { name: 'Choose a model for inference' });
    await expect(well).toBeEnabled();
    await page.screenshot({ path: testInfo.outputPath('inference-well.png') });
    await well.click();
    await expect(page.getByRole('heading', { name: 'Choose model' })).toBeVisible();
});

test('previews recording progress, visible guidance, and a consistent model selector', async ({
    page,
    network,
}, testInfo) => {
    network.use(
        http.get('/api/dataset/{dataset_id}/episodes', () =>
            HttpResponse.json(getMockedEpisodes(Array.from({ length: 24 }, (_, episode_index) => ({ episode_index }))))
        ),
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        )
    );
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.goto(`${RECORD_URL}?summaryPreview=true`);
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
    const recording = page.getByRole('region', { name: 'Recording', exact: true });
    await expect(recording.getByRole('group', { name: '30 episodes in dataset' })).toBeVisible();
    await expect(recording.getByText('+6 this session')).toBeVisible();
    await expect(recording.getByRole('button', { name: 'Recording tips' })).toBeVisible();
    await expect(page.getByRole('region', { name: 'Episode controls' }).getByText(/episodes in dataset/)).toBeHidden();
    await page.screenshot({ path: testInfo.outputPath('progress-and-selector.png') });
    await recording.getByRole('button', { name: 'Recording tips' }).click();
    await expect(page.getByRole('heading', { name: 'Recording a useful dataset' })).toBeVisible();
    await page.getByRole('dialog').evaluate(async (element) => {
        await Promise.allSettled(element.getAnimations({ subtree: true }).map((animation) => animation.finished));
    });
    await page.screenshot({ path: testInfo.outputPath('recording-guidance.png') });
    await page.keyboard.press('Escape');
    await page.getByRole('button', { name: 'Choose a model for inference' }).click();
    await expect(page.getByRole('heading', { name: 'Choose model' })).toBeVisible();
});
