import { describe, expect, it } from 'vitest';

import { buildRobotBody, mergeRobotPayload } from './form-data';

describe('mergeRobotPayload', () => {
    it('preserves SO101 calibration when form-owned fields change', () => {
        const body = buildRobotBody(
            { name: 'Updated arm', connection_string: '/dev/ttyACM1', serial_number: 'SO101-001' },
            'SO101_Follower',
            'robot-1'
        );

        const result = mergeRobotPayload(body, {
            connection_string: '/dev/ttyACM0',
            serial_number: 'SO101-001',
            calibration: {
                shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 10, range_min: -100, range_max: 100 },
            },
        });

        expect(result).not.toBeNull();
        expect(result?.payload).toEqual({
            connection_string: '/dev/ttyACM1',
            serial_number: 'SO101-001',
            calibration: {
                shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 10, range_min: -100, range_max: 100 },
            },
        });
    });

    it('preserves unmodeled bimanual payload fields', () => {
        const body = buildRobotBody(
            {
                name: 'Updated bimanual arm',
                connection_string_left: '192.168.1.2',
                connection_string_right: '192.168.1.3',
                serial_number: '',
            },
            'Trossen_Bimanual_WidowXAI_Follower',
            'robot-1'
        );

        const result = mergeRobotPayload(body, {
            connection_string_left: '192.168.1.1',
            connection_string_right: '192.168.1.4',
            serial_number: '',
            left_calibration: { shoulder_pan: { homing_offset: 10 } },
            right_calibration: { shoulder_pan: { homing_offset: 20 } },
        });

        expect(result).not.toBeNull();
        expect(result?.payload).toEqual({
            connection_string_left: '192.168.1.2',
            connection_string_right: '192.168.1.3',
            serial_number: '',
            left_calibration: { shoulder_pan: { homing_offset: 10 } },
            right_calibration: { shoulder_pan: { homing_offset: 20 } },
        });
    });
});
