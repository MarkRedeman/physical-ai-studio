import { SchemaSessionDataset, SchemaSessionEnvironment, SchemaSessionState } from '../../api/openapi-spec';

export const getMockedSessionEnvironment = (
    overrides: Partial<SchemaSessionEnvironment> = {}
): SchemaSessionEnvironment => ({
    environment_id: 'environment-1',
    name: 'Workbench',
    robot_id: 'robot-1',
    leader_id: null,
    joint_names: [],
    cameras: [],
    ...overrides,
});

export const getMockedSessionDataset = (overrides: Partial<SchemaSessionDataset> = {}): SchemaSessionDataset => ({
    dataset_id: 'dataset-1',
    name: 'Pick and place',
    fps: 30,
    fps_fixed: true,
    is_recording: false,
    episodes_recorded: 0,
    ...overrides,
});

export const getMockedRuntimeSession = (overrides: Partial<SchemaSessionState> = {}): SchemaSessionState => ({
    session_id: 'session-1',
    environment: getMockedSessionEnvironment(),
    dataset: getMockedSessionDataset(),
    model: null,
    control: null,
    ...overrides,
});
