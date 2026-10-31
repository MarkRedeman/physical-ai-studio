import { describe, expect, it } from '@rstest/core';

import { validateRequiredUiFields } from './schema-validation';

const schema = {
    type: 'object',
    properties: {
        id: {
            type: 'string',
            default: '',
            'x-physicalai-ui': { required: true },
        },
        optional: {
            type: 'string',
            default: '',
        },
    },
};

describe('validateRequiredUiFields', () => {
    it.each([undefined, '', '   '])('rejects an empty required string value: %j', (id) => {
        expect(validateRequiredUiFields(schema, { id })).toEqual(['id is required']);
    });

    it('accepts a non-empty required string value', () => {
        expect(validateRequiredUiFields(schema, { id: 'robot-1' })).toEqual([]);
    });

    it('does not change validation for fields without the extension', () => {
        expect(validateRequiredUiFields(schema, { id: 'robot-1', optional: '   ' })).toEqual([]);
    });

    it('checks standard required fields and nested references', () => {
        const nested = {
            properties: { arm: { $ref: '#/$defs/Arm' } },
            required: ['arm'],
            $defs: { Arm: { properties: { port: { type: 'string' } }, required: ['port'] } },
        };
        expect(validateRequiredUiFields(nested, { arm: {} })).toEqual(['arm.port is required']);
        expect(validateRequiredUiFields(nested, { arm: { port: '/dev/fake' } })).toEqual([]);
    });

    it('accepts false, zero, defaults and absent optional objects', () => {
        const fields = {
            properties: {
                enabled: { type: 'boolean' },
                retries: { type: 'integer', default: 0 },
                optional: { properties: { port: { type: 'string' } }, required: ['port'] },
            },
            required: ['enabled', 'retries'],
        };
        expect(validateRequiredUiFields(fields, { enabled: false })).toEqual([]);
    });
});
