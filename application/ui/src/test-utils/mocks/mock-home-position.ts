import { SchemaHomeCapability, SchemaHomePose } from '../../api/openapi-spec';

const names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos'];

export const getMockedHomeCapability = (overrides: Partial<SchemaHomeCapability> = {}): SchemaHomeCapability => ({
    robot_id: 'robot-1',
    fingerprint: 'a'.repeat(64),
    unit: 'normalized',
    joints: Object.fromEntries(
        names.map((name) => [
            name,
            {
                minimum: -100,
                maximum: 100,
                velocity: 5,
                acceleration: 10,
                tolerance: 0.5,
                tracking_error: 2,
            },
        ])
    ),
    preserved_joints: ['gripper.pos'],
    lease_timeout_s: 1.5,
    ...overrides,
});

export const getMockedHomePose = (overrides: Partial<SchemaHomePose> = {}): SchemaHomePose => ({
    version: 1,
    robot_id: 'robot-1',
    fingerprint: 'a'.repeat(64),
    unit: 'normalized',
    positions: Object.fromEntries(names.map((name) => [name, 0])),
    captured_at: '2026-11-04T10:00:00Z',
    ...overrides,
});
