import { useState } from 'react';

import { Button, FileTrigger, Flex, Text, View } from '@geti-ui/ui';

import { InlineAlert } from '../../../setup-wizard/shared/inline-alert';
import { asRecord, resolveReference } from '../schema-utils';
import { FieldSchema } from '../types';

type CalibrationFieldProps = {
    label: string;
    description?: string;
    value: unknown;
    isRequired: boolean;
    onChange: (value: unknown) => void;
    valueSchema?: FieldSchema;
    definitions?: Record<string, FieldSchema>;
};

type CalibrationEntry = {
    id?: unknown;
    drive_mode?: unknown;
    homing_offset?: unknown;
    range_min?: unknown;
    range_max?: unknown;
};

type CalibrationRow = {
    joint: string;
    value: CalibrationEntry;
};

const asCalibrationRows = (value: Record<string, unknown>): CalibrationRow[] =>
    Object.entries(value)
        .map(([joint, entry]) => ({ joint, value: asRecord(entry) }))
        .sort((left, right) => {
            const leftId = typeof left.value.id === 'number' && Number.isFinite(left.value.id) ? left.value.id : Number.POSITIVE_INFINITY;
            const rightId =
                typeof right.value.id === 'number' && Number.isFinite(right.value.id) ? right.value.id : Number.POSITIVE_INFINITY;

            if (leftId !== rightId) {
                return leftId - rightId;
            }
            return left.joint.localeCompare(right.joint);
        });

const formatCell = (value: unknown) => {
    if (typeof value === 'number') {
        return Number.isFinite(value) ? String(value) : '-';
    }
    if (typeof value === 'string' && value !== '') {
        return value;
    }
    return '-';
};

const isExpectedType = (value: unknown, schemaType: string | undefined) => {
    if (schemaType === undefined) {
        return true;
    }
    if (schemaType === 'integer') {
        return typeof value === 'number' && Number.isInteger(value);
    }
    if (schemaType === 'number') {
        return typeof value === 'number' && Number.isFinite(value);
    }
    if (schemaType === 'string') {
        return typeof value === 'string';
    }
    if (schemaType === 'boolean') {
        return typeof value === 'boolean';
    }
    if (schemaType === 'object') {
        return typeof value === 'object' && value !== null && !Array.isArray(value);
    }
    return true;
};

const validateCalibrationEntry = (
    entry: unknown,
    valueSchema: FieldSchema | undefined,
    definitions: Record<string, FieldSchema>
): string | null => {
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
        return 'Each calibration entry must be a JSON object.';
    }

    if (valueSchema === undefined) {
        return null;
    }

    const resolved = resolveReference(valueSchema, definitions);
    const entryRecord = asRecord(entry);
    const required = new Set(resolved.required ?? []);

    for (const requiredField of required) {
        if (entryRecord[requiredField] === undefined || entryRecord[requiredField] === null) {
            return `Calibration entries must include '${requiredField}'.`;
        }
    }

    for (const [name, fieldValue] of Object.entries(entryRecord)) {
        const fieldSchema = resolved.properties?.[name];
        if (fieldSchema === undefined) {
            continue;
        }
        const expectedSchema = resolveReference(fieldSchema, definitions);
        if (!isExpectedType(fieldValue, expectedSchema.type)) {
            return `Calibration field '${name}' has an invalid value type.`;
        }
    }

    return null;
};

const validateCalibrationPayload = (
    value: unknown,
    valueSchema: FieldSchema | undefined,
    definitions: Record<string, FieldSchema>
): string | null => {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        return 'Calibration JSON must be an object keyed by joint name.';
    }

    for (const entry of Object.values(value)) {
        const error = validateCalibrationEntry(entry, valueSchema, definitions);
        if (error !== null) {
            return error;
        }
    }

    return null;
};

export const CalibrationField = ({
    label,
    description,
    value,
    isRequired,
    onChange,
    valueSchema,
    definitions = {},
}: CalibrationFieldProps) => {
    const [error, setError] = useState<string | null>(null);
    const [fileName, setFileName] = useState<string | null>(null);
    const calibration = asRecord(value);
    const rows = asCalibrationRows(calibration);
    const hasCalibration = Object.keys(calibration).length > 0;

    const importCalibration = async (files: FileList | null) => {
        const file = files?.[0] ?? null;
        if (file === null) {
            return;
        }

        setFileName(file.name);
        const text = await file.text();
        let parsed: unknown;
        try {
            parsed = JSON.parse(text);
        } catch {
            setError('Could not parse JSON. Upload a valid calibration .json file.');
            return;
        }

        const validationError = validateCalibrationPayload(parsed, valueSchema, definitions);
        if (validationError !== null) {
            setError(validationError);
            return;
        }

        setError(null);
        onChange(parsed);
    };

    return (
        <Flex direction='column' gap='size-100'>
            <Text
                UNSAFE_style={{
                    fontSize: 'var(--spectrum-global-dimension-font-size-100)',
                    color: 'var(--spectrum-global-color-gray-800)',
                }}
            >
                {label}
                {isRequired ? ' *' : ' (optional)'}
            </Text>
            {description !== undefined && description !== '' && (
                <Text
                    UNSAFE_style={{
                        fontSize: 'var(--spectrum-global-dimension-font-size-100)',
                        color: 'var(--spectrum-global-color-gray-600)',
                    }}
                >
                    {description}
                </Text>
            )}
            <Flex gap='size-100' alignItems='center'>
                <FileTrigger acceptedFileTypes={['.json']} onSelect={importCalibration}>
                    <Button variant='secondary'>{hasCalibration ? 'Replace calibration JSON' : 'Upload calibration JSON'}</Button>
                </FileTrigger>
                {hasCalibration && (
                    <Button
                        variant='secondary'
                        onPress={() => {
                            setError(null);
                            setFileName(null);
                            onChange(null);
                        }}
                    >
                        Clear
                    </Button>
                )}
            </Flex>
            <View>
                <Text>
                    {hasCalibration
                        ? `Loaded ${Object.keys(calibration).length} joint calibration entries${fileName ? ` from ${fileName}` : ''}.`
                        : 'No calibration JSON uploaded.'}
                </Text>
            </View>
            {hasCalibration && (
                <View
                    borderColor='gray-300'
                    borderWidth='thin'
                    backgroundColor='gray-75'
                    padding='size-100'
                    UNSAFE_style={{ borderRadius: 'var(--spectrum-global-dimension-size-50)', overflowX: 'auto' }}
                >
                    <table
                        aria-label='Calibration preview'
                        style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', lineHeight: '16px' }}
                    >
                        <thead>
                            <tr style={{ color: 'var(--spectrum-global-color-gray-700)', textAlign: 'left' }}>
                                <th style={{ padding: '4px 8px', fontWeight: 600 }}>Joint</th>
                                <th style={{ padding: '4px 8px', fontWeight: 600 }}>ID</th>
                                <th style={{ padding: '4px 8px', fontWeight: 600 }}>Drive</th>
                                <th style={{ padding: '4px 8px', fontWeight: 600 }}>Offset</th>
                                <th style={{ padding: '4px 8px', fontWeight: 600 }}>Min</th>
                                <th style={{ padding: '4px 8px', fontWeight: 600 }}>Max</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row) => (
                                <tr key={row.joint} style={{ borderTop: '1px solid var(--spectrum-global-color-gray-300)' }}>
                                    <td style={{ padding: '4px 8px', color: 'var(--spectrum-global-color-gray-800)' }}>
                                        {row.joint}
                                    </td>
                                    <td style={{ padding: '4px 8px' }}>{formatCell(row.value.id)}</td>
                                    <td style={{ padding: '4px 8px' }}>{formatCell(row.value.drive_mode)}</td>
                                    <td style={{ padding: '4px 8px' }}>{formatCell(row.value.homing_offset)}</td>
                                    <td style={{ padding: '4px 8px' }}>{formatCell(row.value.range_min)}</td>
                                    <td style={{ padding: '4px 8px' }}>{formatCell(row.value.range_max)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </View>
            )}
            {error !== null && <InlineAlert variant='error'>{error}</InlineAlert>}
        </Flex>
    );
};
