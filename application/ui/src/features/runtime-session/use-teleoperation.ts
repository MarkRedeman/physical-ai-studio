import { toast } from '@geti-ui/ui';
import { useMutation } from '@tanstack/react-query';

import { SchemaTeleopControlConfig } from '../../api/openapi-spec';
import { ERROR_TOAST, failureOf, useRuntimeSession } from './runtime-session-provider';

/** Read the leader at 100 Hz and ease the follower onto its position on each start. */
export const TELEOP_CONTROL = { type: 'teleop', hz: 100, grace_s: 2 } as const satisfies SchemaTeleopControlConfig;

export interface TeleoperationOptions {
    isDisabled?: boolean;
    onToggle?: (enabled: boolean) => Promise<unknown>;
}

/** Shared by the native switch and its panel so both obey identical readiness/pending guards. */
export const useTeleoperation = ({ isDisabled = false, onToggle }: TeleoperationOptions = {}) => {
    const { state, statuses, send } = useRuntimeSession();
    const { mutate: setTeleop, isPending } = useMutation({
        mutationFn: async (teleoperate: boolean) => {
            if (onToggle != null) {
                await onToggle(teleoperate);
                return;
            }
            const ack = await send({
                command: 'set_control',
                control: teleoperate ? TELEOP_CONTROL : null,
                reject_open_episode: true,
            });
            if (!ack.ok) toast.negative(ack.error?.message ?? 'Could not switch teleoperation.', ERROR_TOAST);
        },
    });
    const disabled =
        isPending ||
        isDisabled ||
        state?.environment == null ||
        failureOf(statuses, 'environment') !== null ||
        failureOf(statuses, 'environment/leader') !== null ||
        failureOf(statuses, 'environment/follower') !== null ||
        (state?.dataset?.is_recording ?? false);

    return {
        isSelected: state?.control?.type === 'teleop',
        isDisabled: disabled,
        onChange: (enabled: boolean) => {
            if (!disabled) setTeleop(enabled);
        },
    };
};
