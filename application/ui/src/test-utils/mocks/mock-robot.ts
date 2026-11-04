import { SchemaSo101FollowerRobotOutput, SchemaSo101LeaderRobotOutput } from '../../api/openapi-spec';

export const getMockedFollowerRobot = (
    overrides: Partial<SchemaSo101FollowerRobotOutput> = {}
): SchemaSo101FollowerRobotOutput => ({
    id: 'robot-1',
    name: 'Workbench follower',
    type: 'SO101_Follower',
    payload: { connection_string: '/dev/ttyACM0', serial_number: '' },
    ...overrides,
});

export const getMockedLeaderRobot = (
    overrides: Partial<SchemaSo101LeaderRobotOutput> = {}
): SchemaSo101LeaderRobotOutput => ({
    id: 'leader-1',
    name: 'Workbench leader',
    type: 'SO101_Leader',
    payload: { connection_string: '/dev/ttyACM1', serial_number: '' },
    ...overrides,
});
