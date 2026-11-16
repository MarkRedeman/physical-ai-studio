import { useEffect, useState } from 'react';

import { Flex, ProgressCircle, Switch, toast, View } from '@geti-ui/ui';
import { useMutation } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import { getRobotConnectionErrorTitle } from '../../../api/errors';
import { SchemaTeleopControlConfig } from '../../../api/openapi-spec';
import { useProjectId } from '../../projects/use-project';
import { RobotViewer, UnavailableRobotViewer } from '../../robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { AvailableSchemaRobot, isUnavailableRobot } from '../../robots/robot-types';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { useSynchronizeModelJoints } from '../../robots/use-synchronize-model-joints';
import { ERROR_TOAST, failureOf, useRuntimeSession, useSessionJoints } from '../runtime-session-provider';

/** Default teleoperation settings; force feedback requires explicit opt-in. */
export const TELEOP_CONTROL = {
    type: 'teleop',
    hz: 100,
    grace_s: 2,
    force_feedback_gain: 0,
} as const satisfies SchemaTeleopControlConfig;

/** Drive the follower from the leader, or hold it where it is. */
export const TeleopSwitch = () => {
    const { state, statuses, send } = useRuntimeSession();
    const [forceFeedback, setForceFeedback] = useState(false);
    useEffect(() => setForceFeedback(false), [state?.environment?.leader_id]);
    const teleop = state?.control?.type === 'teleop' ? state.control : undefined;
    const isTeleop = teleop !== undefined;
    const { mutate: setTeleop, isPending } = useMutation({
        mutationFn: async (teleoperate: boolean) => {
            const ack = await send({
                command: 'set_control',
                control: teleoperate ? { ...TELEOP_CONTROL, force_feedback_gain: forceFeedback ? 0.1 : 0 } : null,
            });
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? 'Could not switch teleoperation.', ERROR_TOAST);
            }
        },
    });

    return (
        <Flex direction='column'>
            <Switch
                isEmphasized
                isSelected={isTeleop}
                // A failed environment stays loaded until it is loaded again, with nothing to drive;
                // switching the control mid-episode discards the episode.
                isDisabled={isPending || failureOf(statuses, 'environment') !== null || state?.dataset?.is_recording}
                onChange={setTeleop}
            >
                Teleoperate
            </Switch>
            <Switch
                isSelected={teleop !== undefined ? teleop.force_feedback_gain > 0 : forceFeedback}
                isDisabled={isTeleop || isPending || state?.dataset?.is_recording}
                onChange={setForceFeedback}
            >
                Force feedback (compatible leaders)
            </Switch>
        </Flex>
    );
};

const LiveRobot = ({ robot }: { robot: AvailableSchemaRobot }) => {
    const joints = useSessionJoints();
    useSynchronizeModelJoints(joints, robot);

    return <RobotViewer robot={robot} />;
};

/** The session's follower. ``showTeleop`` puts the Teleoperate switch on it, for pages without a side bar. */
export const RobotCell = ({ follower_id, showTeleop = true }: { follower_id: string; showTeleop?: boolean }) => {
    const { project_id } = useProjectId();
    const { state, error, isLoading } = useRuntimeSession();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id: follower_id } },
    });
    if (isUnavailableRobot(robot)) {
        return <UnavailableRobotViewer robotType={robot.type} />;
    }

    if (error !== null && !isLoading) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <InlineAlert variant='error'>
                    <strong>{getRobotConnectionErrorTitle(error.error_code)}</strong>
                    <br />
                    {error.message}
                </InlineAlert>
            </Flex>
        );
    }

    if (state?.environment?.robot_id !== follower_id) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <ProgressCircle isIndeterminate aria-label='Connecting robot' />
            </Flex>
        );
    }

    return (
        <RobotModelsProvider>
            <View
                minWidth='size-4000'
                minHeight='size-4000'
                width='100%'
                height='100%'
                backgroundColor={'gray-600'}
                position={'relative'}
            >
                <LiveRobot robot={robot} />
                {showTeleop && state.environment.leader_id !== null && (
                    <View position={'absolute'} right={0} top={0} padding='size-100'>
                        <TeleopSwitch />
                    </View>
                )}
            </View>
        </RobotModelsProvider>
    );
};
