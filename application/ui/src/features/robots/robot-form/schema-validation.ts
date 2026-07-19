import { FieldSchema } from './schema-fields';

type JsonSchema = {
    properties?: Record<string, FieldSchema>;
    $defs?: Record<string, FieldSchema>;
};

const asRecord = (value: unknown): Record<string, unknown> =>
    typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const resolveReference = (schema: FieldSchema, definitions: Record<string, FieldSchema>): FieldSchema => {
    if (schema.$ref === undefined) return schema;

    const definition = definitions[schema.$ref.replace('#/$defs/', '')];
    return definition === undefined ? schema : { ...definition, ...schema };
};

const isMissingRequiredValue = (value: unknown, schema: FieldSchema) =>
    value === undefined ||
    value === null ||
    (schema.type === 'string' && typeof value === 'string' && value.trim() === '');

export const validateRequiredUiFields = (schema: JsonSchema, payload: Record<string, unknown>): string[] => {
    const definitions = schema.$defs ?? {};

    const validateProperties = (
        properties: Record<string, FieldSchema> | undefined,
        values: Record<string, unknown>,
        prefix = ''
    ): string[] =>
        Object.entries(properties ?? {}).flatMap(([name, field]) => {
            const resolved = resolveReference(field, definitions);
            const fieldName = `${prefix}${name}`;
            const value = values[name];

            if (resolved['x-physicalai-ui']?.required && isMissingRequiredValue(value, resolved)) {
                return [`${fieldName} is required`];
            }

            return resolved.properties === undefined
                ? []
                : validateProperties(resolved.properties, asRecord(value), `${fieldName}.`);
        });

    return validateProperties(schema.properties, payload);
};
