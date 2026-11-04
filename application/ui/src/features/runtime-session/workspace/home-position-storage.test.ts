import { getMockedHomeCapability, getMockedHomePose } from '../../../test-utils/mocks/mock-home-position';
import { getMockedSessionEnvironment } from '../../../test-utils/mocks/mock-runtime-session';
import { decodeStoredHome, homeStorageKey, validateHomePose } from './home-position-storage';

it('restores only a compatible pose without inventing coordinates', () => {
    const pose = getMockedHomePose();
    expect(decodeStoredHome(JSON.stringify(pose), getMockedHomeCapability())).toEqual(pose);
    expect(decodeStoredHome('{broken', getMockedHomeCapability())).toBeNull();
    expect(decodeStoredHome(null, getMockedHomeCapability())).toBeNull();
});

it.each([
    { robot_id: 'another-robot' },
    { fingerprint: 'b'.repeat(64) },
    { unit: 'degrees' },
    { version: 2 },
    { positions: {} },
    { positions: { 'shoulder_pan.pos': 500 } },
    { captured_at: 'invalid date' },
    { unknown: true },
])('rejects invalid or incompatible saved home data: %j', (update) => {
    expect(validateHomePose({ ...getMockedHomePose(), ...update }, getMockedHomeCapability())).toBeNull();
});

it('rejects nonfinite, coerced, or extra gripper coordinates', () => {
    const pose = getMockedHomePose();
    for (const value of [NaN, Infinity, '20', true]) {
        expect(
            validateHomePose(
                { ...pose, positions: { ...pose.positions, 'shoulder_pan.pos': value } },
                getMockedHomeCapability()
            )
        ).toBeNull();
    }
    expect(
        validateHomePose({ ...pose, positions: { ...pose.positions, 'gripper.pos': 0 } }, getMockedHomeCapability())
    ).toBeNull();
});

it('scopes storage to the project, environment and follower; unsaved environments are session-only', () => {
    const environment = getMockedSessionEnvironment();
    expect(homeStorageKey('project-1', environment)).not.toBe(homeStorageKey('project-2', environment));
    expect(homeStorageKey('project-1', environment)).not.toBe(
        homeStorageKey('project-1', { ...environment, robot_id: 'other' })
    );
    expect(homeStorageKey('project-1', { ...environment, environment_id: null })).toBeNull();
});
