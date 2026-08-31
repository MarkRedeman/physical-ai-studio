import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { beforeEach, describe, expect, it } from 'vitest';

import { SchemaStreamingSettings } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { createQueryClient } from '../../../query-client/query-client';
import { render } from '../../../test-utils/render';
import { StreamingSettingsForm } from './streaming-settings-form';

const cpuH264Preset: SchemaStreamingSettings = {
    vcodec: 'libx264',
    pix_fmt: null,
    crf: 23,
    preset: 'veryfast',
    encoder_threads: null,
    encoder_queue_maxsize: 60,
};

const renderForm = (streaming: SchemaStreamingSettings = cpuH264Preset) =>
    render(<StreamingSettingsForm streaming={streaming} />);

beforeEach(() => {
    server.use(
        http.get('/api/system/devices/training', () =>
            HttpResponse.json({ mode: 'local', remote_available: true, devices: [] })
        )
    );
});

describe('StreamingSettingsForm', () => {
    it('shows a summary and Customize button when a preset is selected', () => {
        renderForm();

        expect(screen.getByText('libx264')).toBeInTheDocument();
        expect(screen.getByText('veryfast')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Customize' })).toBeInTheDocument();
        expect(screen.queryByLabelText('Video codec')).not.toBeInTheDocument();
    });

    it('reveals editable fields prefilled with the preset values after Customize', async () => {
        const user = userEvent.setup();
        renderForm();

        await user.click(screen.getByRole('button', { name: 'Customize' }));

        expect(screen.getByLabelText('Video codec')).toHaveValue('libx264');
        expect(screen.getByLabelText('CRF')).toHaveValue('23');
        expect(screen.getByLabelText('Preset')).toHaveValue('veryfast');
        expect(screen.queryByRole('button', { name: 'Customize' })).not.toBeInTheDocument();
    });

    it('shows editable fields when the values do not match any preset', () => {
        renderForm({ ...cpuH264Preset, crf: 30 });

        expect(screen.getByLabelText('Video codec')).toHaveValue('libx264');
        expect(screen.getByLabelText('CRF')).toHaveValue('30');
        expect(screen.queryByRole('button', { name: 'Customize' })).not.toBeInTheDocument();
    });

    it('shows a summary when switching to a preset from the configuration picker', async () => {
        const user = userEvent.setup();
        renderForm({ ...cpuH264Preset, crf: 30 });

        await user.click(screen.getByRole('button', { name: /Configuration/ }));
        await user.click(await screen.findByRole('option', { name: /CPU H\.264/ }));

        expect(screen.getByText('23')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Customize' })).toBeInTheDocument();
    });

    it('marks the recommended preset in the picker instead of showing an apply button', async () => {
        const queryClient = createQueryClient();
        queryClient.setQueryData(['get', '/api/system/devices/training'], {
            mode: 'local',
            remote_available: true,
            devices: [{ type: 'xpu', name: 'Intel GPU' }],
        });
        const user = userEvent.setup();
        render(<StreamingSettingsForm streaming={cpuH264Preset} />, { queryClient });

        await user.click(screen.getByRole('button', { name: /Configuration/ }));

        const recommendedOption = await screen.findByRole('option', { name: /Intel GPU QSV/ });
        expect(recommendedOption).toHaveTextContent('Recommended');
        expect(screen.queryByRole('button', { name: 'Apply recommended preset' })).not.toBeInTheDocument();
    });
});
