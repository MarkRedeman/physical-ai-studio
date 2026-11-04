import { Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaSessionEnvironment, SchemaSessionState, SchemaStatusMessage } from '../../../api/openapi-spec';
import RobotArm from '../../../assets/robot-arm.webp';
import { useProjectId } from '../../projects/use-project';
import { isUnavailableRobot } from '../../robots/robot-types';
import { Statuses } from '../runtime-session-provider';

import classes from './environment-robots.module.css';

interface RobotCardProps {
    robot_id: string;
    robotRole: 'Follower' | 'Leader';
    status: SchemaStatusMessage | undefined;
    isActive: boolean;
    environmentFailed: boolean;
}

const RobotCard = ({ robot_id, robotRole, status, isActive, environmentFailed }: RobotCardProps) => {
    const { project_id } = useProjectId();
    const { data: robot, isError } = $api.useQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id } },
    });
    const getStatus = () => {
        if (status?.state === 'failed' || status?.state === 'stopped') return { label: 'Disconnected', tone: 'error' };
        if (status?.state === 'starting') return { label: 'Connecting…', tone: 'neutral' };
        if (environmentFailed) return { label: 'Not ready', tone: 'neutral' };
        // A running device is authoritative even if its plugin metadata is no longer available.
        if (status?.state === 'running')
            return isActive ? { label: 'Active', tone: 'active' } : { label: 'Ready', tone: 'neutral' };
        if (robot && isUnavailableRobot(robot)) return { label: 'Unavailable', tone: 'error' };
        return { label: 'Waiting', tone: 'neutral' };
    };
    const activity = getStatus();

    return (
        <div className={classes.robotCard} data-state={activity.tone} role='group' aria-label={`${robotRole} robot`}>
            <div className={classes.robotSummary}>
                <img src={RobotArm} alt='' className={classes.thumbnail} />
                <div className={classes.robotCopy}>
                    <Text UNSAFE_className={classes.robotName}>{robot?.name ?? `${robotRole} robot`}</Text>
                    <Text UNSAFE_className={classes.robotRole}>{robotRole}</Text>
                </div>
                <span className={classes.activity} data-state={activity.tone} role='status'>
                    {activity.label}
                </span>
            </div>
            {status?.state === 'failed' && status.error && <Text UNSAFE_className={classes.error}>{status.error}</Text>}
            {isError && <Text UNSAFE_className={classes.robotRole}>Robot details could not be loaded.</Text>}
        </div>
    );
};

/** Runtime status reports readiness; the selected control reports which robots are active. */
export const EnvironmentRobots = ({
    environment,
    statuses,
    control,
}: {
    environment: SchemaSessionEnvironment;
    statuses: Statuses;
    control: SchemaSessionState['control'];
}) => {
    const environmentFailed = statuses.environment?.state === 'failed';
    const controlFailed = statuses.control?.state === 'failed';
    const isTeleop = control?.type === 'teleop' && !controlFailed;
    const isModel = control?.type === 'model' && !controlFailed;

    return (
        <div className={classes.robots}>
            <RobotCard
                robot_id={environment.robot_id}
                robotRole='Follower'
                status={statuses['environment/follower']}
                isActive={isTeleop || isModel}
                environmentFailed={environmentFailed}
            />
            {environment.leader_id != null && (
                <RobotCard
                    robot_id={environment.leader_id}
                    robotRole='Leader'
                    status={statuses['environment/leader']}
                    isActive={isTeleop}
                    environmentFailed={environmentFailed}
                />
            )}
        </div>
    );
};
