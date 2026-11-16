import { useReducer } from 'react';

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { SchemaTeleopControlConfig } from '../../../api/openapi-spec';
import { render } from '../../../test-utils/render';
import { TeleopSwitch } from './robot-cell';

const send = rs.fn().mockResolvedValue({ ok: true });
let control: SchemaTeleopControlConfig | null = null;
let leaderId = 'leader-1';
let isRecording = false;

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: {},
    failureOf: () => null,
    useRuntimeSession: () => ({
        state: { control, environment: { leader_id: leaderId }, dataset: { is_recording: isRecording } },
        statuses: {},
        send,
    }),
}));

describe('TeleopSwitch force feedback', () => {
    beforeEach(() => {
        send.mockClear();
        control = null;
        leaderId = 'leader-1';
        isRecording = false;
    });

    it('starts ordinary teleoperation with feedback disabled', async () => {
        const user = userEvent.setup();
        render(<TeleopSwitch />);
        expect(screen.getByRole('switch', { name: /force feedback/i })).not.toBeChecked();
        await user.click(screen.getByRole('switch', { name: 'Teleoperate' }));
        await waitFor(() =>
            expect(send).toHaveBeenCalledWith({
                command: 'set_control',
                control: { type: 'teleop', hz: 100, grace_s: 2, force_feedback_gain: 0 },
            })
        );
    });

    it('enables a conservative gain only after the operator opts in', async () => {
        const user = userEvent.setup();
        render(<TeleopSwitch />);
        await user.click(screen.getByRole('switch', { name: /force feedback/i }));
        await user.click(screen.getByRole('switch', { name: 'Teleoperate' }));
        await waitFor(() =>
            expect(send).toHaveBeenCalledWith({
                command: 'set_control',
                control: { type: 'teleop', hz: 100, grace_s: 2, force_feedback_gain: 0.1 },
            })
        );
    });

    it('shows the running config and requires stopping before changing feedback', async () => {
        control = { type: 'teleop', hz: 100, grace_s: 2, force_feedback_gain: 0.1 };
        const user = userEvent.setup();
        render(<TeleopSwitch />);
        const feedback = screen.getByRole('switch', { name: /force feedback/i });
        expect(feedback).toBeChecked();
        expect(feedback).toBeDisabled();
        await user.click(screen.getByRole('switch', { name: 'Teleoperate' }));
        await waitFor(() => expect(send).toHaveBeenCalledWith({ command: 'set_control', control: null }));
    });

    it('does not carry feedback consent to a different leader', async () => {
        const user = userEvent.setup();
        const Reloadable = () => {
            const [, refresh] = useReducer((value: number) => value + 1, 0);
            return (
                <>
                    <button onClick={refresh}>Refresh environment</button>
                    <TeleopSwitch />
                </>
            );
        };
        render(<Reloadable />);
        await user.click(screen.getByRole('switch', { name: /force feedback/i }));
        leaderId = 'leader-2';
        await user.click(screen.getByRole('button', { name: 'Refresh environment' }));
        expect(screen.getByRole('switch', { name: /force feedback/i })).not.toBeChecked();
    });

    it('does not change feedback during a recording', () => {
        isRecording = true;
        render(<TeleopSwitch />);
        expect(screen.getByRole('switch', { name: /force feedback/i })).toBeDisabled();
    });
});
