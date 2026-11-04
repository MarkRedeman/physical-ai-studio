import { ReactNode } from 'react';

import { Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaSessionEnvironment, SchemaSessionState, SchemaStatusMessage } from '../../../api/openapi-spec';
import { ReactComponent as CameraIcon } from '../../../assets/camera.svg';
import RobotArm from '../../../assets/robot-arm.webp';
import { useProjectId } from '../../projects/use-project';
import { isUnavailableRobot } from '../../robots/robot-types';
import { Statuses } from '../runtime-session-provider';

import classes from './environment-devices.module.css';

interface DeviceActivity {
    label: string;
    tone: 'neutral' | 'active' | 'error';
}

const deviceActivity = (
    status: SchemaStatusMessage | undefined,
    isActive: boolean,
    environmentFailed: boolean,
    activeLabel = 'Active'
): DeviceActivity => {
    if (status?.state === 'failed' || status?.state === 'stopped') return { label: 'Disconnected', tone: 'error' };
    if (status?.state === 'starting') return { label: 'Connecting…', tone: 'neutral' };
    if (environmentFailed) return { label: 'Not ready', tone: 'neutral' };
    if (status?.state === 'running')
        return isActive ? { label: activeLabel, tone: 'active' } : { label: 'Ready', tone: 'neutral' };
    return { label: 'Waiting', tone: 'neutral' };
};

const DeviceCard = ({
    name,
    subtitle,
    kind,
    icon,
    activity,
    error,
    note,
    accessibleName,
}: {
    name: string;
    subtitle: string;
    kind: 'follower' | 'leader' | 'camera';
    icon: ReactNode;
    activity: DeviceActivity;
    error?: string | null;
    note?: string;
    accessibleName: string;
}) => (
    <div className={classes.deviceCard} data-state={activity.tone} role='group' aria-label={accessibleName}>
        <div className={classes.deviceSummary}>
            <div className={classes.thumbnailTile} data-kind={kind} aria-hidden='true'>
                {icon}
            </div>
            <div className={classes.deviceCopy}>
                <Text UNSAFE_className={classes.deviceName}>{name}</Text>
                <Text UNSAFE_className={classes.deviceRole}>{subtitle}</Text>
            </div>
            <span className={classes.activity} data-state={activity.tone} role='status'>
                {activity.label}
            </span>
        </div>
        {error && <Text UNSAFE_className={classes.error}>{error}</Text>}
        {note && <Text UNSAFE_className={classes.deviceRole}>{note}</Text>}
    </div>
);

const RobotCard = ({
    robot_id,
    robotRole,
    status,
    isActive,
    environmentFailed,
}: {
    robot_id: string;
    robotRole: 'Follower' | 'Leader';
    status: SchemaStatusMessage | undefined;
    isActive: boolean;
    environmentFailed: boolean;
}) => {
    const { project_id } = useProjectId();
    const { data: robot, isError } = $api.useQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id } },
    });
    let activity = deviceActivity(status, isActive, environmentFailed);
    // A running device is authoritative even if its plugin metadata is no longer available.
    if (robot && isUnavailableRobot(robot) && status?.state !== 'running' && status?.state !== 'failed') {
        activity = { label: 'Unavailable', tone: 'error' };
    }

    return (
        <DeviceCard
            name={robot?.name ?? `${robotRole} robot`}
            subtitle={robotRole}
            kind={robotRole === 'Follower' ? 'follower' : 'leader'}
            icon={<img src={RobotArm} alt='' className={classes.thumbnail} />}
            activity={activity}
            accessibleName={`${robotRole} robot`}
            error={status?.state === 'failed' ? status.error : undefined}
            note={isError ? 'Robot details could not be loaded.' : undefined}
        />
    );
};

/** Runtime status reports readiness; control state distinguishes active robots from ready ones. */
export const EnvironmentDevices = ({
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
        <div className={classes.devices}>
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
            {environment.cameras.map((camera) => {
                const status = statuses[`environment/${camera.key}`];
                return (
                    <DeviceCard
                        key={camera.camera_id}
                        name={camera.name}
                        subtitle={`Camera · ${camera.width} × ${camera.height}`}
                        kind='camera'
                        icon={<CameraIcon className={classes.cameraIcon} />}
                        accessibleName={`Camera: ${camera.name}`}
                        activity={deviceActivity(status, true, environmentFailed, 'Streaming')}
                        error={status?.state === 'failed' ? status.error : undefined}
                    />
                );
            })}
        </div>
    );
};
