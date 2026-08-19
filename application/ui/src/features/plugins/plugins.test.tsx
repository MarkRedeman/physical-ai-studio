import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { PluginsView } from './plugins';

const installedPlugin = {
    id: 'physicalai-rebot-b601-plugin',
    name: 'ReBot Plugin',
    description: 'ReBot B601 and Arm102 robot integrations.',
    category: 'ReBot',
    source: 'first_party',
    repo_url: 'https://github.com/example/rebot',
    installed: true,
    installed_version: '0.1.0',
    in_use_robot_count: 0,
    robots: [
        { type: 'ReBot_B601_DM_Follower', display_name: 'ReBot B601 DM Follower', role: 'follower', installed: true },
    ],
};

const availablePlugin = {
    id: 'physicalai-mujoco-so101-plugin',
    name: 'MuJoCo Plugin',
    description: 'MuJoCo-backed SO-101 simulation integration.',
    category: 'MuJoCo',
    source: 'first_party',
    repo_url: 'https://github.com/example/mujoco',
    installed: false,
    installed_version: null,
    in_use_robot_count: 0,
    robots: [{ type: 'MuJoCo_SO101_Follower', display_name: 'MuJoCo SO101 Follower', role: 'follower', installed: false }],
};

describe('PluginsView', () => {
    it('renders installed and available plugin sections', async () => {
        server.use(http.get('/api/plugins', () => HttpResponse.json([installedPlugin, availablePlugin])));

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        expect(await screen.findByRole('heading', { name: 'Plugins' })).toBeVisible();
        expect(screen.getByRole('heading', { name: 'Installed' })).toBeVisible();
        expect(screen.getByRole('heading', { name: 'Available' })).toBeVisible();
        expect(screen.getByText('ReBot Plugin')).toBeVisible();
        expect(screen.getByText('MuJoCo Plugin')).toBeVisible();
        expect(screen.getByText('ReBot B601 DM Follower')).toBeVisible();
    });

    it('shows a restart-required banner after installing a plugin', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin])),
            http.post('/api/plugins/{plugin_id}/install', () =>
                HttpResponse.json({ restart_required: true })
            )
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await user.click(await screen.findByRole('button', { name: 'Install' }));

        expect(
            await screen.findByText('Restart the server to activate the plugin changes.')
        ).toBeVisible();
        expect(screen.getByRole('button', { name: 'Restart server' })).toBeVisible();
    });

    it('disables uninstall for plugins with robots in use', async () => {
        server.use(
            http.get('/api/plugins', () =>
                HttpResponse.json([{ ...installedPlugin, in_use_robot_count: 2 }])
            )
        );

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        const uninstallButton = await screen.findByRole('button', { name: 'Uninstall' });
        expect(uninstallButton).toBeDisabled();
        expect(screen.getByText(/In use by 2 robots/)).toBeVisible();
    });
});
