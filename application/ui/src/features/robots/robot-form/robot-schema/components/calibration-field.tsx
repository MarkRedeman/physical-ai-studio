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
            <Text>
                {label}
                {isRequired ? ' *' : ''}
            </Text>
            {description !== undefined && description !== '' && <Text>{description}</Text>}
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
            {error !== null && <InlineAlert variant='error'>{error}</InlineAlert>}
        </Flex>
    );
};
