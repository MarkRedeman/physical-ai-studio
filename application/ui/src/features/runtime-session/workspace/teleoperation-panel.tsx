import { Switch, Text } from '@geti-ui/ui';

import { TELEOP_CONTROL, useTeleoperation } from '../use-teleoperation';
import { useSurfaceActivation } from './mode-card-interaction';
import { usePartControls } from './part-controls';
import { useSessionMode } from './session-mode';

import classes from './session-workspace.module.css';

export const TeleoperationPanel = () => {
    const { mode, isPending, setTeleoperation } = useSessionMode();
    const { pendingPart } = usePartControls();
    const control = useTeleoperation({
        isDisabled: isPending || pendingPart !== null,
        onToggle: (enabled) => setTeleoperation(enabled ? TELEOP_CONTROL : null),
    });
    const surfaceEvents = useSurfaceActivation(() => control.onChange(!control.isSelected), control.isDisabled);

    return (
        <div
            className={classes.teleop}
            role='group'
            aria-label='Teleoperation'
            data-active={control.isSelected || undefined}
            data-toggleable={!control.isDisabled || undefined}
            {...surfaceEvents}
        >
            <Switch isEmphasized {...control}>
                Teleoperate
            </Switch>
            <Text UNSAFE_className={classes.hint}>
                {mode !== 'record'
                    ? 'Enable teleoperation to stop model control and switch to Recording.'
                    : control.isSelected
                      ? 'The follower tracks your leader robot.'
                      : 'Enable to guide the follower with your leader robot.'}
            </Text>
        </div>
    );
};
