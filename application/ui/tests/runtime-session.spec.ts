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
let controlCommandCount = 0;

test.beforeEach(async ({ network, page }) => {
    controlCommandCount = 0;
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
        if (page.url().includes('emptyPreview')) {
            state.environment = null;
            state.dataset = null;
        }
        if (page.url().includes('modePreview'))
            state.model = {
                model_id: 'model-1',
                name: 'Cube policy',
                backend: new URL(page.url()).searchParams.get('backend') ?? 'onnx',
                device: 'cpu',
                variant: 'original',
                rtc_enabled: false,
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
            if (command.command === 'set_control') {
                controlCommandCount += 1;
                state.control = command.control;
            }
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

test('previews mode radios and moves the selection border without starting robot motion', async ({
    page,
    network,
}, testInfo) => {
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

test('distinguishes neutral borders, selection, hover, and keyboard focus', async ({ page }) => {
    await page.goto(RECORD_URL);
    const environment = page.getByRole('region', { name: 'Environment' });
    const recording = page.getByRole('region', { name: 'Recording', exact: true });
    const inference = page.getByRole('region', { name: 'Inference', exact: true });
    await expect(page.getByText(/Once you train a model/)).toBeVisible();
    await expect(environment).toHaveCSS('border-top-color', 'rgb(69, 88, 115)');
    await expect(inference).toHaveCSS('border-top-color', 'rgb(69, 88, 115)');
    await expect(inference).toHaveCSS('border-top-style', 'dashed');
    await expect(recording).toHaveCSS('border-top-color', 'rgb(75, 156, 245)');
    await expect(recording).toHaveCSS('box-shadow', 'none');
    await expect(page.getByRole('region', { name: 'Episode controls' })).toHaveCSS(
        'border-top-color',
        'rgb(54, 68, 89)'
    );
    await inference.hover();
    await expect(inference).toHaveCSS('border-top-color', 'rgb(92, 117, 153)');
    await page.getByRole('button', { name: 'Back to dataset' }).focus();
    await page.getByRole('radio', { name: 'Recording', exact: true }).focus();
    await page.keyboard.press('ArrowDown');
    await expect(inference).toHaveCSS('border-top-color', 'rgb(75, 156, 245)');
    await expect(inference).toHaveCSS('outline-style', 'solid');
    await expect(inference).toHaveCSS('outline-width', '2px');
    expect(controlCommandCount).toBe(0);
});

test('uses matching panel headings and keeps guidance larger than metadata', async ({ page }) => {
    await page.goto(RECORD_URL);
    for (const name of ['Environment', 'Recording', 'Inference']) {
        const heading = page.getByRole('heading', { name, exact: true });
        await expect(heading).toHaveCSS('font-size', '16px');
        await expect(heading).toHaveCSS('font-weight', '600');
        await expect(heading).toHaveCSS('color', 'rgb(255, 255, 255)');
    }
    await expect(page.getByText(/Once you train a model/)).toHaveCSS('font-size', '14px');
    await expect(page.getByText('Pick and place', { exact: true })).toHaveCSS('font-weight', '600');
    await expect(page.getByText('Recording at 30 fps · fixed for this dataset')).toHaveCSS('font-size', '12px');
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

test('spans both workspace columns with recording and inference controls', async ({ page, network }) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        ),
        http.get('/api/models/{model_id}', () => HttpResponse.json({ model: getMockedModel(), exports: [] }))
    );
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.goto(`${RECORD_URL}?modePreview=true`);
    for (const mode of ['record', 'inference']) {
        if (mode === 'inference') await page.getByRole('radio', { name: 'Inference', exact: true }).click();
        const controls = await page
            .getByRole('region', { name: mode === 'record' ? 'Episode controls' : 'Inference controls' })
            .boundingBox();
        const live = await page.getByRole('main', { name: 'Live session' }).boundingBox();
        const setup = await page.getByRole('complementary', { name: 'Session setup' }).boundingBox();
        expect(controls).not.toBeNull();
        expect(live).not.toBeNull();
        expect(setup).not.toBeNull();
        expect(controls?.x).toBe(live?.x);
        expect((controls?.x ?? 0) + (controls?.width ?? 0)).toBe((setup?.x ?? 0) + (setup?.width ?? 0));
        expect(live?.y).toBeGreaterThan((controls?.y ?? 0) + (controls?.height ?? 0));
        expect(setup?.y).toBe(live?.y);
        for (const name of ['Environment', 'Recording', 'Inference']) {
            const panel = await page.getByRole('region', { name, exact: true }).boundingBox();
            expect(panel?.x).toBe(setup?.x);
            expect(panel?.width).toBe(setup?.width);
            if (name === 'Environment') expect(panel?.y).toBe(live?.y);
        }
    }
    expect(controlCommandCount).toBe(0);
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
                            'Move the red block to the left side of the table, ' +
                                'then return the arm to its starting position.',
                        ],
                    },
                ])
            )
        )
    );
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(RECORD_URL);
    await expect(page.getByRole('textbox', { name: /Current task/ })).toBeVisible();
    const historyButton = page.getByRole('button', { name: 'Use previous task…' });
    const historyBounds = await historyButton.boundingBox();
    expect(historyBounds?.height).toBeLessThanOrEqual(24);
    expect(historyBounds?.width).toBeLessThanOrEqual(24);
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

test('previews the model header action', async ({ page, network }, testInfo) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        )
    );
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(RECORD_URL);
    const choose = page.getByRole('button', { name: 'Choose model…' });
    await expect(choose).toBeEnabled();
    await page.screenshot({ path: testInfo.outputPath('inference-selector.png') });
    await choose.click();
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
    const summaryBounds = await recording.getByRole('region', { name: 'Dataset progress' }).boundingBox();
    const datasetBounds = await recording.getByText('Pick and place', { exact: true }).boundingBox();
    const rateBounds = await recording.getByText('Recording at 30 fps · fixed for this dataset').boundingBox();
    const tipsBounds = await recording.getByRole('button', { name: 'Recording tips' }).boundingBox();
    expect((summaryBounds?.y ?? 0) + (summaryBounds?.height ?? 0)).toBeLessThan(datasetBounds?.y ?? 0);
    expect(rateBounds?.y).toBeGreaterThan(datasetBounds?.y ?? 0);
    expect(tipsBounds?.height).toBeLessThanOrEqual(24);
    await expect(page.getByRole('region', { name: 'Episode controls' }).getByText(/episodes in dataset/)).toBeHidden();
    await page.screenshot({ path: testInfo.outputPath('progress-and-selector.png') });
    await recording.getByRole('button', { name: 'Recording tips' }).click();
    await expect(page.getByRole('heading', { name: 'Recording a useful dataset' })).toBeVisible();
    await page.getByRole('dialog').evaluate(async (element) => {
        await Promise.allSettled(element.getAnimations({ subtree: true }).map((animation) => animation.finished));
    });
    await page.screenshot({ path: testInfo.outputPath('recording-guidance.png') });
    await page.keyboard.press('Escape');
    await page.getByRole('button', { name: 'Choose model…' }).click();
    await expect(page.getByRole('heading', { name: 'Choose model' })).toBeVisible();
});

test('previews backend branding and compact inference execution settings', async ({ page, network }, testInfo) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['openvino'] })])
        ),
        http.get('/api/models/{model_id}', () => HttpResponse.json({ model: getMockedModel(), exports: [] }))
    );
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.goto(`${RECORD_URL}?modePreview=true&backend=openvino`);
    await page.getByRole('radio', { name: 'Inference', exact: true }).click();
    const inference = page.getByRole('region', { name: 'Inference', exact: true });
    await expect(inference.getByText('OpenVINO · original · cpu')).toBeVisible();
    const syncBounds = await inference.getByRole('radio', { name: 'Synchronous', exact: true }).boundingBox();
    const asyncBounds = await inference.getByRole('radio', { name: 'Asynchronous', exact: true }).boundingBox();
    expect(asyncBounds?.y).toBe(syncBounds?.y);
    expect(asyncBounds?.x).toBeGreaterThan(syncBounds?.x ?? 0);
    await expect(inference.getByRole('slider', { name: 'Request threshold' })).toBeHidden();
    const logoBounds = await inference.locator('[class*="backendLogo"] svg').boundingBox();
    expect(logoBounds?.width).toBeGreaterThan(40);
    expect(logoBounds?.height).toBeGreaterThan(40);
    await inference.getByRole('button', { name: /Execution settings/ }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath('inference-compact.png') });
    await inference.getByRole('button', { name: /Execution settings/ }).click();
    await expect(inference.getByRole('slider', { name: 'Request threshold' })).toBeVisible();
    await expect(inference.getByRole('slider', { name: 'Action rate' })).toBeVisible();
    await inference.getByRole('radio', { name: 'Asynchronous', exact: true }).click();
    await expect(inference.getByRole('slider', { name: 'Request threshold' })).toBeVisible();
    await inference.getByRole('slider', { name: 'Request threshold' }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath('inference-advanced.png') });
});

test('selects modes from card backgrounds and titles while respecting episode locks', async ({ page, network }) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        ),
        http.get('/api/models/{model_id}', () => HttpResponse.json({ model: getMockedModel(), exports: [] }))
    );
    await page.setViewportSize({ width: 1440, height: 1400 });
    await page.goto(`${RECORD_URL}?modePreview=true`);
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
    const recording = page.getByRole('region', { name: 'Recording', exact: true });
    const inference = page.getByRole('region', { name: 'Inference', exact: true });
    await inference.click({ position: { x: 8, y: 8 } });
    await expect(page.getByRole('radio', { name: 'Inference', exact: true })).toBeChecked();
    expect(controlCommandCount).toBe(0);
    await recording.getByRole('heading', { name: 'Recording', exact: true }).click();
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeChecked();
    await page.getByRole('radio', { name: 'Inference', exact: true }).focus();
    await page.keyboard.press('Space');
    await expect(page.getByRole('radio', { name: 'Inference', exact: true })).toBeChecked();
    expect(controlCommandCount).toBe(0);
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeVisible();
    await recording.click({ position: { x: 8, y: 8 } });
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeChecked();
    expect(controlCommandCount).toBe(2); // Start once, then hold once when deliberately switching cards.
    await page.getByRole('button', { name: /Start episode/ }).click();
    await expect(page.getByRole('button', { name: /Save episode/ })).toBeVisible();
    await inference.click({ position: { x: 8, y: 8 } });
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeChecked();
    await expect(page.getByRole('radio', { name: 'Inference', exact: true })).toBeDisabled();
    expect(controlCommandCount).toBe(2);
});

test('keeps nested inference fields and recording slider tracks independent of card selection', async ({
    page,
    network,
}) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        ),
        http.get('/api/models/{model_id}', () => HttpResponse.json({ model: getMockedModel(), exports: [] }))
    );
    await page.setViewportSize({ width: 1440, height: 1450 });
    await page.goto(`${RECORD_URL}?modePreview=true&newDataset=true`);
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
    const recording = page.getByRole('region', { name: 'Recording', exact: true });
    const inference = page.getByRole('region', { name: 'Inference', exact: true });
    await inference.getByRole('textbox', { name: 'Task', exact: true }).click();
    await inference.getByRole('textbox', { name: 'Task', exact: true }).fill('Move the cube');
    await inference.getByRole('radio', { name: 'Asynchronous', exact: true }).click();
    await inference.getByRole('button', { name: /Execution settings/ }).click();
    await expect(inference.getByRole('slider', { name: 'Request threshold' })).toBeVisible();
    await inference.getByRole('button', { name: 'About inference latency' }).locator('svg').click();
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeChecked();
    expect(controlCommandCount).toBe(0);
    await page.getByRole('radio', { name: 'Inference', exact: true }).click();
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeVisible();
    const rate = recording.getByRole('slider', { name: 'Recording rate' });
    const field = recording
        .locator('[data-mode-interactive]')
        .filter({ has: page.getByRole('slider', { name: 'Recording rate' }) });
    const fieldBounds = await field.boundingBox();
    const sliderBounds = await rate.boundingBox();
    expect(fieldBounds).not.toBeNull();
    expect(sliderBounds).not.toBeNull();
    const valueBefore = await rate.evaluate(
        (element) =>
            element.getAttribute('aria-valuenow') ?? (element instanceof HTMLInputElement ? element.value : null)
    );
    await field.click({
        position: {
            x: (fieldBounds?.width ?? 0) * 0.8,
            y: (sliderBounds?.y ?? 0) - (fieldBounds?.y ?? 0) + (sliderBounds?.height ?? 0) / 2,
        },
    });
    await expect
        .poll(() =>
            rate.evaluate(
                (element) =>
                    element.getAttribute('aria-valuenow') ??
                    (element instanceof HTMLInputElement ? element.value : null)
            )
        )
        .not.toBe(valueBefore);
    await expect(page.getByRole('radio', { name: 'Inference', exact: true })).toBeChecked();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeVisible();
    expect(controlCommandCount).toBe(1);
});

test('does not stop inference when editing recording fields or using their menus and portaled dialogs', async ({
    page,
    network,
}) => {
    const previousTask = 'Pick the cube from the tray';
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        ),
        http.get('/api/models/{model_id}', () => HttpResponse.json({ model: getMockedModel(), exports: [] })),
        http.get('/api/dataset/{dataset_id}/episodes', () =>
            HttpResponse.json(getMockedEpisodes([{ tasks: [previousTask] }]))
        )
    );
    await page.setViewportSize({ width: 1440, height: 1450 });
    await page.goto(`${RECORD_URL}?modePreview=true`);
    await page.getByRole('radio', { name: 'Inference', exact: true }).click();
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeVisible();
    const recording = page.getByRole('region', { name: 'Recording', exact: true });
    const task = recording.getByRole('textbox', { name: 'Current task' });
    await task.click();
    await task.fill('A new multiline task\nwith more instructions');
    await recording.locator('label').filter({ hasText: 'Current task' }).click();
    await expect(task).toBeFocused();
    const taskBounds = await task.boundingBox();
    const cardBounds = await recording.boundingBox();
    expect(taskBounds).not.toBeNull();
    expect(cardBounds).not.toBeNull();
    await page.mouse.move(
        (taskBounds?.x ?? 0) + (taskBounds?.width ?? 0) / 2,
        (taskBounds?.y ?? 0) + (taskBounds?.height ?? 0) / 2
    );
    await page.mouse.down();
    await page.mouse.move((cardBounds?.x ?? 0) + 8, (cardBounds?.y ?? 0) + 8, { steps: 5 });
    await page.mouse.up();
    await expect(page.getByRole('radio', { name: 'Inference', exact: true })).toBeChecked();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeVisible();
    await recording.getByRole('button', { name: 'Use previous task…' }).locator('svg').click();
    await page.getByRole('searchbox', { name: 'Find a task' }).fill('cube');
    await page.getByRole('button', { name: previousTask, exact: true }).click();
    await expect(task).toHaveValue(previousTask);
    await recording.getByRole('button', { name: 'Recording tips' }).click();
    await page
        .getByRole('dialog')
        .getByText(/starting recommendation/)
        .click();
    await expect(page.getByRole('radio', { name: 'Inference', exact: true })).toBeChecked();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
    await recording.getByRole('button', { name: 'dataset actions' }).click();
    await page.getByRole('menuitem', { name: 'Change dataset…' }).click();
    await expect(page.getByRole('heading', { name: 'Choose dataset' })).toBeVisible();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(page.getByRole('radio', { name: 'Inference', exact: true })).toBeChecked();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeVisible();
    expect(controlCommandCount).toBe(1);
});

test('opens the model chooser without selecting its card', async ({ page, network }) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        )
    );
    await page.goto(RECORD_URL);
    await page.getByRole('button', { name: 'Choose model…' }).click();
    await expect(page.getByRole('heading', { name: 'Choose model' })).toBeVisible();
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeChecked();
    expect(controlCommandCount).toBe(0);
});

test('keeps empty resource names quiet and disabled model guidance readable', async ({ page, network }, testInfo) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        )
    );
    await page.goto(`${RECORD_URL}?emptyPreview=true`);
    for (const resource of ['environment', 'dataset', 'model']) {
        const emptyName = page.getByText(`No ${resource} selected`, { exact: true });
        await expect(emptyName).toHaveCSS('font-size', '14px');
        await expect(emptyName).toHaveCSS('font-weight', '400');
        await expect(emptyName).toHaveCSS('color', 'rgb(143, 162, 188)');
    }
    await expect(page.getByRole('button', { name: 'Choose model…' })).toBeDisabled();
    const guidance = page.getByText('Connect an environment first');
    await expect(guidance).toBeVisible();
    await expect(guidance).toHaveCSS('font-size', '14px');
    await expect(guidance).toHaveCSS('color', 'rgb(143, 162, 188)');
    expect(await guidance.evaluate((element) => element.closest('button'))).toBeNull();
    await page.screenshot({ path: testInfo.outputPath('empty-resources.png') });
});

test('toggles teleoperation from its panel without double-toggling the switch or label', async ({ page }) => {
    await page.goto(RECORD_URL);
    const panel = page.getByRole('group', { name: 'Teleoperation', exact: true });
    const toggle = panel.getByRole('switch', { name: 'Teleoperate' });
    await expect(toggle).toBeEnabled();
    await panel.click({ position: { x: 8, y: 8 } });
    await expect(toggle).toBeChecked();
    await expect(toggle).toBeEnabled();
    expect(controlCommandCount).toBe(1);
    await panel.click({ position: { x: 8, y: 8 } });
    await expect(toggle).not.toBeChecked();
    await expect(toggle).toBeEnabled();
    expect(controlCommandCount).toBe(2);
    await toggle.click();
    await expect(toggle).toBeChecked();
    await expect(toggle).toBeEnabled();
    expect(controlCommandCount).toBe(3);
    const label = panel.getByText('Teleoperate', { exact: true });
    await label.scrollIntoViewIfNeeded();
    const labelBounds = await label.boundingBox();
    expect(labelBounds).not.toBeNull();
    // Spectrum overlays a transparent input on the label; click where a user clicks, not the covered span.
    await page.mouse.click(
        (labelBounds?.x ?? 0) + (labelBounds?.width ?? 0) / 2,
        (labelBounds?.y ?? 0) + (labelBounds?.height ?? 0) / 2
    );
    await expect(toggle).not.toBeChecked();
    await expect(toggle).toBeEnabled();
    expect(controlCommandCount).toBe(4);
    await toggle.focus();
    await page.keyboard.press('Space');
    await expect(toggle).toBeChecked();
    await expect(toggle).toBeEnabled();
    expect(controlCommandCount).toBe(5);
    await page.getByRole('button', { name: /Start episode/ }).click();
    await expect(toggle).toBeDisabled();
    await panel.click({ position: { x: 8, y: 8 } });
    await expect(toggle).toBeChecked();
    expect(controlCommandCount).toBe(5);
});

test('does not add a panel toggle to a drag that starts on the native switch', async ({ page }) => {
    await page.goto(RECORD_URL);
    const panel = page.getByRole('group', { name: 'Teleoperation', exact: true });
    const toggle = panel.getByRole('switch', { name: 'Teleoperate' });
    await expect(toggle).toBeEnabled();
    const toggleBounds = await toggle.boundingBox();
    const panelBounds = await panel.boundingBox();
    expect(toggleBounds).not.toBeNull();
    expect(panelBounds).not.toBeNull();
    await page.mouse.move(
        (toggleBounds?.x ?? 0) + (toggleBounds?.width ?? 0) / 2,
        (toggleBounds?.y ?? 0) + (toggleBounds?.height ?? 0) / 2
    );
    await page.mouse.down();
    await page.mouse.move((panelBounds?.x ?? 0) + 8, (panelBounds?.y ?? 0) + 8, { steps: 5 });
    await page.mouse.up();
    // The native switch may complete its own press; the background must never add a second toggle.
    expect(controlCommandCount).toBeLessThanOrEqual(1);
});

test('does not toggle teleoperation from the panel when a device is disconnected', async ({ page }) => {
    await page.goto(`${RECORD_URL}?deviceFailure=true`);
    const panel = page.getByRole('group', { name: 'Teleoperation', exact: true });
    await expect(panel.getByRole('switch', { name: 'Teleoperate' })).toBeDisabled();
    await panel.click({ position: { x: 8, y: 8 } });
    await expect(panel.getByRole('switch', { name: 'Teleoperate' })).not.toBeChecked();
    expect(controlCommandCount).toBe(0);
});

test('enables teleoperation from the panel when inference is active and selects Recording', async ({
    page,
    network,
}) => {
    network.use(
        http.get('/api/projects/{project_id}/models', () =>
            HttpResponse.json([getMockedModel({ available_backends: ['onnx'] })])
        ),
        http.get('/api/models/{model_id}', () => HttpResponse.json({ model: getMockedModel(), exports: [] }))
    );
    await page.goto(`${RECORD_URL}?modePreview=true`);
    await page.getByRole('radio', { name: 'Inference', exact: true }).click();
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeVisible();
    const panel = page.getByRole('group', { name: 'Teleoperation', exact: true });
    await expect(panel.getByRole('switch', { name: 'Teleoperate' })).toBeEnabled();
    await panel.click({ position: { x: 8, y: 8 } });
    await expect(panel.getByRole('switch', { name: 'Teleoperate' })).toBeChecked();
    await expect(page.getByRole('radio', { name: 'Recording', exact: true })).toBeChecked();
    await expect(page.getByRole('button', { name: 'Stop model', exact: true })).toBeHidden();
    expect(controlCommandCount).toBe(2);
});
