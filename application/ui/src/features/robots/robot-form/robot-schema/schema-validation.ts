import { asRecord, resolveReference } from './schema-utils';
import { FieldSchema, JsonSchema } from './types';

const isFieldUiOptions = (value: FieldSchema['x-physicalai-ui']): value is { required?: boolean } =>
    value !== undefined && !Array.isArray(value);

const isMissingRequiredValue = (value: unknown, schema: FieldSchema) =>
    value === undefined ||
    value === null ||
    (schema.type === 'string' && typeof value === 'string' && value.trim() === '');

export const validateRequiredUiFields = (schema: JsonSchema, payload: Record<string, unknown>): string[] => {
    const definitions = schema.$defs ?? {};

    const validateProperties = (
        properties: Record<string, FieldSchema> | undefined,
        values: Record<string, unknown>,
        required: string[] = [],
        prefix = ''
    ): string[] =>
        Object.entries(properties ?? {}).flatMap(([name, field]) => {
            const resolved = resolveReference(field, definitions);
            const fieldName = `${prefix}${name}`;
            const value = values[name] === undefined ? resolved.default : values[name];
            const requiredField =
                required.includes(name) ||
                (isFieldUiOptions(resolved['x-physicalai-ui']) && resolved['x-physicalai-ui'].required);

            if (requiredField && isMissingRequiredValue(value, resolved)) {
                return [`${fieldName} is required`];
            }

            return resolved.properties === undefined || value === null || (value === undefined && !requiredField)
                ? []
                : validateProperties(resolved.properties, asRecord(value), resolved.required, `${fieldName}.`);
        });

    return validateProperties(schema.properties, payload, schema.required);
};
