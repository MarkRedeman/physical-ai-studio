import { HttpResponse } from 'msw';

import { getMockedDataset } from '../src/test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../src/test-utils/mocks/mock-environment';
import { getMockedProject } from '../src/test-utils/mocks/mock-project';
import { getMockedRuntimeSession } from '../src/test-utils/mocks/mock-runtime-session';
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
        http.get('/api/projects/{project_id}/robots/{robot_id}', () =>
            HttpResponse.json({
                id: 'robot-1',
                name: 'Demo follower',
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
        const publish = () => socket.send(JSON.stringify({ type: 'state', data: state }));
        socket.onMessage((message) => {
            const command = JSON.parse(String(message));
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
    await expect(page.getByText('Plugin unavailable:')).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('ready.png') });
});

test('keeps episode actions visible and locks resource changes while recording', async ({ page }, testInfo) => {
    await page.goto(RECORD_URL);
    await page.getByRole('button', { name: /Start episode/ }).click();
    await expect(page.getByRole('button', { name: /Save episode/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'dataset actions' })).toBeDisabled();
    await expect(page.getByRole('textbox', { name: /Task/ })).toHaveAttribute('readonly');
    await page.screenshot({ path: testInfo.outputPath('recording.png') });
    await page.getByRole('button', { name: /Save episode/ }).click();
    await expect(page.getByRole('button', { name: /Start episode/ })).toBeEnabled();
    await expect(page.getByText(/1 saved this session/)).toBeVisible();
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
