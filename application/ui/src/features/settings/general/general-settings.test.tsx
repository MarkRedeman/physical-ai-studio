import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaUserSettingsResponse } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { GeneralSettings } from './general-settings';

const SETTINGS_PATH = '/api/settings';
const DEVICES_PATH = '/api/system/devices/training';

const settings: SchemaUserSettingsResponse = {
    geti_action_dataset_path: '/data/physicalai/datasets',
    streaming: {
        vcodec: 'libx264',
        pix_fmt: 'nv12',
        crf: 23,
        preset: 'veryfast',
        encoder_threads: 4,
        encoder_queue_maxsize: 60,
    },
    trainer: {
        request_timeout_s: 30,
        download_read_timeout_s: 120,
        stream_reconnect_max_s: 900,
        stream_reconnect_backoff_max_s: 30,
    },
    huggingface: { hf_token: '**********' },
    logger: { providers: ['csv'], wandb_project: null, wandb_entity: null, wandb_api_key: '**********' },
};

// Streaming config that exactly matches the CPU · H.264 preset.
const presetSettings: SchemaUserSettingsResponse = {
    ...settings,
    streaming: {
        vcodec: 'libx264',
        pix_fmt: null,
        crf: 23,
        preset: 'veryfast',
        encoder_threads: null,
        encoder_queue_maxsize: 60,
    },
};

const withinSection = (title: string) => {
    const heading = screen.getByRole('heading', { name: title });
    const section = heading.parentElement;
    if (section === null) throw new Error(`Section ${title} has no parent`);
    return within(section);
};

describe('GeneralSettings', () => {
    beforeEach(() => {
        server.use(http.get(SETTINGS_PATH, () => HttpResponse.json(settings)));
    });

    it('renders all four settings sections with their current values', async () => {
        render(<GeneralSettings />);

        expect(await screen.findByRole('heading', { name: 'General' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Streaming' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Trainer' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Hugging Face' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Logging' })).toBeInTheDocument();

        expect(screen.getByRole('textbox', { name: 'Video codec' })).toHaveValue('libx264');
        expect(screen.getByRole('textbox', { name: 'Request timeout (s)' })).toHaveValue('30');
        expect(screen.getByText(/Datasets directory/)).toBeInTheDocument();
    });

    it('saves only the streaming group when its Save button is pressed', async () => {
        const user = userEvent.setup();
        let captured: Record<string, unknown> | undefined;
        server.use(
            http.patch(SETTINGS_PATH, async ({ request }) => {
                captured = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(settings);
            })
        );

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Streaming');
        const save = section.getByRole('button', { name: 'Save' });
        expect(save).toBeDisabled();

        await user.clear(section.getByRole('textbox', { name: 'Video codec' }));
        await user.type(section.getByRole('textbox', { name: 'Video codec' }), 'libsvtav1');
        await user.click(section.getByRole('button', { name: 'Save' }));

        await screen.findByText('Saved');
        expect(captured).toEqual({
            streaming: {
                vcodec: 'libsvtav1',
                pix_fmt: 'nv12',
                crf: 23,
                preset: 'veryfast',
                encoder_threads: 4,
                encoder_queue_maxsize: 60,
            },
        });
    });

    it('fills fields from a preset, makes them read-only, and saves the preset values', async () => {
        const user = userEvent.setup();
        let captured: Record<string, unknown> | undefined;
        server.use(
            http.patch(SETTINGS_PATH, async ({ request }) => {
                captured = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(settings);
            })
        );

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Streaming');
        await user.click(section.getByRole('button', { name: /configuration/i }));
        await user.click(await screen.findByRole('option', { name: /cpu · h\.265/i }));

        const codec = section.getByRole('textbox', { name: 'Video codec' });
        expect(codec).toHaveValue('libx265');
        expect(codec).toHaveAttribute('readonly');

        await user.click(section.getByRole('button', { name: 'Save' }));

        await screen.findByText('Saved');
        expect(captured).toEqual({
            streaming: {
                vcodec: 'libx265',
                pix_fmt: null,
                crf: 28,
                preset: 'veryfast',
                encoder_threads: null,
                encoder_queue_maxsize: 60,
            },
        });
    });

    it('detects a matching preset on load and keeps the fields read-only', async () => {
        server.use(http.get(SETTINGS_PATH, () => HttpResponse.json(presetSettings)));

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Streaming');
        expect(section.getByRole('button', { name: /configuration/i })).toHaveTextContent(/cpu · h\.264/i);
        expect(section.getByRole('textbox', { name: 'Video codec' })).toHaveAttribute('readonly');
    });

    it('recommends an NVIDIA preset from detected training devices and applies it', async () => {
        const user = userEvent.setup();
        server.use(
            http.get(DEVICES_PATH, () =>
                HttpResponse.json({
                    mode: 'local',
                    remote_available: true,
                    devices: [{ type: 'cuda', name: 'NVIDIA GeForce RTX 4090', memory: 24_000_000_000, index: 0 }],
                })
            )
        );

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Streaming');
        expect(
            await screen.findByText((content) => content.includes('we recommend NVIDIA GPU · NVENC'))
        ).toBeInTheDocument();

        await user.click(section.getByRole('button', { name: /apply/i }));

        const codec = section.getByRole('textbox', { name: 'Video codec' });
        expect(codec).toHaveValue('h264_nvenc');
        expect(codec).toHaveAttribute('readonly');
    });

    it('recommends an Intel preset when an XPU accelerator is present', async () => {
        server.use(
            http.get(DEVICES_PATH, () =>
                HttpResponse.json({
                    mode: 'local',
                    remote_available: true,
                    devices: [{ type: 'xpu', name: 'Intel Arc A770', memory: 16_000_000_000, index: 0 }],
                })
            )
        );

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        expect(
            await screen.findByText((content) => content.includes('we recommend Intel GPU · QSV'))
        ).toBeInTheDocument();
    });

    it('saves only the trainer group independently', async () => {
        const user = userEvent.setup();
        let captured: Record<string, unknown> | undefined;
        server.use(
            http.patch(SETTINGS_PATH, async ({ request }) => {
                captured = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(settings);
            })
        );

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Trainer');
        await user.clear(section.getByRole('textbox', { name: 'Request timeout (s)' }));
        await user.type(section.getByRole('textbox', { name: 'Request timeout (s)' }), '45');
        await user.click(section.getByRole('button', { name: 'Save' }));

        await screen.findByText('Saved');
        expect(captured).toEqual({
            trainer: {
                request_timeout_s: 45,
                download_read_timeout_s: 120,
                stream_reconnect_max_s: 900,
                stream_reconnect_backoff_max_s: 30,
            },
        });
    });

    it('sets a new Hugging Face token and omits it when untouched', async () => {
        const user = userEvent.setup();
        const captured: Record<string, unknown>[] = [];
        server.use(
            http.patch(SETTINGS_PATH, async ({ request }) => {
                captured.push((await request.json()) as Record<string, unknown>);
                return HttpResponse.json(settings);
            })
        );

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Hugging Face');
        expect(section.getByRole('button', { name: 'Save' })).toBeDisabled();

        await user.type(section.getByLabelText('Hugging Face token'), 'hf-new-token');
        await user.click(section.getByRole('button', { name: 'Save' }));

        await screen.findByText('Saved');
        expect(captured[0]).toEqual({ huggingface: { hf_token: 'hf-new-token' } });
    });

    it('sends an explicit null when the configured token is marked for removal', async () => {
        const user = userEvent.setup();
        let captured: Record<string, unknown> | undefined;
        server.use(
            http.patch(SETTINGS_PATH, async ({ request }) => {
                captured = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(settings);
            })
        );

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Hugging Face');
        await user.click(section.getByRole('switch', { name: /remove the configured value/i }));
        await user.click(section.getByRole('button', { name: 'Save' }));

        await screen.findByText('Saved');
        expect(captured).toEqual({ huggingface: { hf_token: null } });
    });

    it('saves logger providers and never echoes back an unchanged API key', async () => {
        const user = userEvent.setup();
        let captured: Record<string, unknown> | undefined;
        server.use(
            http.patch(SETTINGS_PATH, async ({ request }) => {
                captured = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(settings);
            })
        );

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Logging');
        await user.click(section.getByRole('switch', { name: /weights & biases/i }));
        await user.click(section.getByRole('button', { name: 'Save' }));

        await screen.findByText('Saved');
        expect(captured).toEqual({
            logger: {
                providers: ['csv', 'wandb'],
                wandb_project: null,
                wandb_entity: null,
            },
        });
    });

    it('keeps W&B settings visible but disabled until W&B is enabled', async () => {
        const user = userEvent.setup();

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Logging');
        const project = section.getByRole('textbox', { name: 'W&B project' });
        expect(project).toBeDisabled();

        await user.click(section.getByRole('switch', { name: /weights & biases/i }));

        expect(section.getByRole('textbox', { name: 'W&B project' })).toBeEnabled();
    });

    it('drops a disabled logger from providers on save', async () => {
        const user = userEvent.setup();
        let captured: Record<string, unknown> | undefined;
        server.use(
            http.patch(SETTINGS_PATH, async ({ request }) => {
                captured = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(settings);
            })
        );

        render(<GeneralSettings />);
        await screen.findByRole('heading', { name: 'General' });

        const section = withinSection('Logging');
        await user.click(section.getByRole('switch', { name: /csv logger/i }));
        await user.click(section.getByRole('button', { name: 'Save' }));

        await screen.findByText('Saved');
        expect(captured).toEqual({
            logger: {
                providers: [],
                wandb_project: null,
                wandb_entity: null,
            },
        });
    });
});
