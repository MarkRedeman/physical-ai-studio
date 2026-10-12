import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { SchemaStreamingSettings } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { StreamingSettingsForm } from './streaming-settings-form';

const streaming: SchemaStreamingSettings = { profile: null };

const renderForm = (settings: SchemaStreamingSettings = streaming) => {
    server.use(
        http.get('/api/system/streaming-profiles', () =>
            HttpResponse.json({
                profiles: [
                    {
                        id: 'intel-qsv-h264',
                        name: 'Intel GPU QSV',
                        description: 'Hardware H.264 encoding using Intel Quick Sync Video.',
                    },
                ],
            })
        )
    );
    return render(<StreamingSettingsForm streaming={settings} />);
};

describe('StreamingSettingsForm', () => {
    it('offers Default and the profiles verified by the backend', async () => {
        renderForm();

        expect(await screen.findByRole('button', { name: /Configuration/ })).toHaveTextContent('Default');
        await userEvent.setup().click(screen.getByRole('button', { name: /Configuration/ }));
        expect(await screen.findByRole('option', { name: /Intel GPU QSV/ })).toBeInTheDocument();
    });

    it('saves an explicit backend profile', async () => {
        let requestBody: unknown;
        server.use(
            http.patch('/api/settings', async ({ request }) => {
                requestBody = await request.json();
                return HttpResponse.json({
                    streaming: { profile: 'intel-qsv-h264' },
                    trainer: {
                        request_timeout_s: 30,
                        download_read_timeout_s: 120,
                        stream_reconnect_max_s: 900,
                        stream_reconnect_backoff_max_s: 30,
                    },
                    huggingface: { hf_token: null },
                    ssh: {
                        connect_timeout_s: 10,
                        command_timeout_s: 15,
                        preflight_timeout_s: 30,
                        image_pull_timeout_s: 1800,
                        trainer_shm_size_gb: 32,
                    },
                    hotkeys: { bindings: {} },
                });
            })
        );
        const user = userEvent.setup();
        renderForm();

        await user.click(await screen.findByRole('button', { name: /Configuration/ }));
        await user.click(await screen.findByRole('option', { name: /Intel GPU QSV/ }));
        await user.click(screen.getByRole('button', { name: 'Save' }));

        expect(requestBody).toEqual({ streaming: { profile: 'intel-qsv-h264' } });
    });
});
