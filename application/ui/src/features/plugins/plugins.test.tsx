import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it, vi } from 'vitest';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { PluginsView } from './plugins';

const installedPlugin = {
    id: 'physicalai-rebot-b601-plugin',
    name: 'ReBot Plugin',
    description: 'ReBot B601 and Arm102 robot integrations.',
    repo_url: 'https://github.com/example/rebot',
    installed: true,
    installed_version: '0.1.0',
    in_use_robot_count: 0,
    robots: [
        {
            type: 'ReBot_B601_DM_Follower',
            display_name: 'ReBot B601 DM Follower',
            role: 'follower' as const,
            installed: true,
        },
    ],
};

const availablePlugin = {
    id: 'physicalai-mujoco-so101-plugin',
    name: 'MuJoCo Plugin',
    description: 'MuJoCo-backed SO-101 simulation integration.',
    repo_url: 'https://github.com/example/mujoco',
    installed: false,
    installed_version: null,
    in_use_robot_count: 0,
    robots: [
        {
            type: 'MuJoCo_SO101_Follower',
            display_name: 'MuJoCo SO101 Follower',
            role: 'follower' as const,
            installed: false,
        },
    ],
};

const lerobotPlugin = {
    id: 'physicalai-lerobot-plugin',
    name: 'LeRobot Plugin',
    description: 'Robot and teleoperator configurations discovered from LeRobot.',
    repo_url: 'https://github.com/example/lerobot',
    installed: true,
    installed_version: '0.1.0',
    in_use_robot_count: 0,
    robots: [],
};

describe('PluginsView', () => {
    it('renders installed and available plugins in a single table', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([installedPlugin, availablePlugin, lerobotPlugin]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        const rebotRow = await screen.findByTestId('plugin-row-physicalai-rebot-b601-plugin');
        await user.click(within(rebotRow).getByText('ReBot Plugin'));

        expect(await screen.findByRole('heading', { name: 'Plugins' })).toBeVisible();
        expect(screen.getByText('Plugin')).toBeVisible();
        expect(screen.getAllByText('Robots')).toHaveLength(2);
        expect(screen.getByText('ReBot Plugin')).toBeVisible();
        expect(screen.getByText('MuJoCo Plugin')).toBeVisible();
        expect(screen.getByRole('heading', { name: 'Robots' })).toBeVisible();
        expect(screen.getByText('ReBot B601 DM Follower')).toBeVisible();
        expect(screen.getAllByText('1 robot')).toHaveLength(2);
    });

    it('opens a restart prompt after installing a plugin', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin])),
            http.post('/api/plugins/{plugin_id}', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await user.click(await screen.findByRole('button', { name: 'Install' }));

        expect(await screen.findByText('Plugin changes require a server restart to become active.')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Restart now' })).toBeVisible();
        await user.click(screen.getByRole('button', { name: 'Later' }));
    });

    it('restores missing plugins and opens the restart prompt', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin])),
            http.get('/api/plugins/restore-status', () =>
                HttpResponse.json({
                    needs_restore: true,
                    missing_plugin_ids: [availablePlugin.id],
                    unknown_plugin_ids: [],
                })
            ),
            http.post('/api/plugins:restore', () =>
                HttpResponse.json({
                    restored_plugin_ids: [availablePlugin.id],
                    failed_plugin_ids: [],
                    unknown_plugin_ids: [],
                    restart_required: true,
                })
            ),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        expect(
            await screen.findByText(
                (_, element) => element?.textContent === '1 previously installed plugin need restoration.'
            )
        ).toBeVisible();
        await user.click(screen.getByRole('button', { name: 'Restore plugins' }));

        expect(await screen.findByText('Plugin changes require a server restart to become active.')).toBeVisible();
        await user.click(screen.getByRole('button', { name: 'Later' }));
    });

    it('restarts after confirming the restart prompt', async () => {
        let restartCalls = 0;
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin])),
            http.post('/api/plugins/{plugin_id}', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([])),
            http.post('/api/system/restart', () => {
                restartCalls += 1;
                return HttpResponse.json({ status: 'restarting' });
            }),
            http.get('/api/health', () => {
                return HttpResponse.json({
                    status: 'healthy',
                    instance_id: restartCalls === 0 ? 'before-restart' : 'after-restart',
                    restart_required: restartCalls === 0,
                });
            })
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await user.click(await screen.findByRole('button', { name: 'Install' }));
        await user.click(await screen.findByRole('button', { name: 'Restart now' }));

        await vi.waitFor(() => {
            expect(restartCalls).toBe(1);
        });

        expect(await screen.findByText('Waiting for server restart…')).toBeVisible();
    });

    it('disables uninstall for plugins with robots in use', async () => {
        server.use(http.get('/api/plugins', () => HttpResponse.json([{ ...installedPlugin, in_use_robot_count: 2 }])));
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        const rebotRow = await screen.findByTestId('plugin-row-physicalai-rebot-b601-plugin');
        await user.click(within(rebotRow).getByText('ReBot Plugin'));

        const uninstallButton = await screen.findByRole('button', { name: 'Uninstall' });
        expect(uninstallButton).toBeDisabled();
        expect(screen.getByText(/In use by 2 robots/)).toBeVisible();
    });

    it('opens the restart prompt after uninstalling a plugin', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([installedPlugin])),
            http.delete('/api/plugins/{plugin_id}', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await user.click(await screen.findByRole('button', { name: 'Uninstall' }));

        expect(await screen.findByText('Plugin changes require a server restart to become active.')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Restart now' })).toBeVisible();
        await user.click(screen.getByRole('button', { name: 'Later' }));
    });
});
