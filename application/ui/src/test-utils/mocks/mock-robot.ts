import { SchemaRobotCreateResponse } from '../../features/robots/robot-types';

type TrossenRobot = Extract<SchemaRobotCreateResponse, { type: 'Trossen_WidowXAI_Follower' }>;

export const getMockedTrossenRobot = (overrides: Partial<TrossenRobot> = {}): TrossenRobot => ({
    id: 'robot-1',
    name: 'Arm',
    type: 'Trossen_WidowXAI_Follower',
    payload: { connection_string: '192.0.2.1' },
    ...overrides,
});
