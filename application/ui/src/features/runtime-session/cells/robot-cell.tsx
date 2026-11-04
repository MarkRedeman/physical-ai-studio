import { Flex, ProgressCircle, Switch, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getRobotConnectionErrorTitle } from '../../../api/errors';
import { useProjectId } from '../../projects/use-project';
import { RobotViewer, UnavailableRobotViewer } from '../../robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { AvailableSchemaRobot, isUnavailableRobot } from '../../robots/robot-types';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { useSynchronizeModelJoints } from '../../robots/use-synchronize-model-joints';
import { useRuntimeSession, useSessionJoints } from '../runtime-session-provider';
import { TeleoperationOptions, useTeleoperation } from '../use-teleoperation';

export { TELEOP_CONTROL } from '../use-teleoperation';

/** Drive the follower from the leader, or hold it where it is. */
export const TeleopSwitch = ({ isDisabled = false, onToggle }: TeleoperationOptions) => {
    const control = useTeleoperation({ isDisabled, onToggle });
    return (
        <Switch isEmphasized {...control}>
            Teleoperate
        </Switch>
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
