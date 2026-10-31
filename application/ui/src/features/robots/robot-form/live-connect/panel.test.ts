import { describe, expect, it } from '@rstest/core';
import { URDFRobot } from 'urdf-loader';

import { jointPresentation } from './panel';

const model = (jointType: string, lower: number, upper: number) =>
    ({ joints: { joint: { jointType, limit: { lower, upper } } } }) as unknown as URDFRobot;
const jointMap = { 'shoulder.pos': ['joint'], 'gripper.pos': ['joint'] };

describe('live joint presentation', () => {
    it('keeps normalized SO101 values distinct from angles', () => {
        const result = jointPresentation(
            'SO101_Follower',
            'normalized',
            'shoulder.pos',
            50,
            model('revolute', -2, 2),
            jointMap
        );
        expect(result.unit).toBe('normalized');
        expect(result.range).toEqual([-100, 100]);
        expect(result.modelValue).toBe(1);
    });

    it('maps gripper normalized values into its model range', () => {
        const result = jointPresentation(
            'SO101_Leader',
            'normalized',
            'gripper.pos',
            50,
            model('revolute', 0, 1),
            jointMap
        );
        expect(result.range).toEqual([0, 100]);
        expect(result.modelValue).toBe(0.5);
    });

    it('converts angular limits once and preserves out-of-range readings', () => {
        const result = jointPresentation(
            'Trossen_WidowXAI_Follower',
            undefined,
            'shoulder.pos',
            120,
            model('revolute', -Math.PI / 2, Math.PI / 2),
            jointMap
        );
        expect(result.range).toEqual([-90, 90]);
        expect(result.value).toBe(120);
        expect(result.modelValue).toBeCloseTo((2 * Math.PI) / 3);
    });

    it('keeps prismatic coordinates in meters', () => {
        const result = jointPresentation(
            'Trossen_WidowXAI_Follower',
            undefined,
            'gripper.pos',
            0.01,
            model('prismatic', 0, 0.04),
            jointMap
        );
        expect(result.unit).toBe('m');
        expect(result.modelValue).toBe(0.01);
        expect(result.range).toEqual([0, 0.04]);
    });

    it('does not invent ranges for continuous joints', () => {
        expect(
            jointPresentation(
                'Trossen_WidowXAI_Follower',
                undefined,
                'shoulder.pos',
                90,
                model('continuous', 0, 0),
                jointMap
            ).range
        ).toBeNull();
    });

    it('does not guess units or animate unknown plugins', () => {
        const result = jointPresentation(
            'External_Follower',
            undefined,
            'shoulder.pos',
            1000,
            model('revolute', -1, 1),
            jointMap
        );
        expect(result).toEqual({ value: 1000, unit: 'raw', range: null, modelValue: null });
    });
});
