import { useEffect, useState } from 'react';

import { Flex, Heading, Switch, View } from '@geti-ui/ui';

import { ConnectionField, ConnectionGroupOptions } from './connection-field';
import { useRobotForm } from './provider';
import { FieldSchema, SchemaField } from './schema-fields';

type JsonSchema = {
    type?: string;
    properties?: Record<string, FieldSchema>;
    required?: string[];
    $defs?: Record<string, FieldSchema>;
    ['x-physicalai-ui']?: { groups?: Record<string, GroupOptions> };
};
type GroupOptions = ConnectionGroupOptions & {
    title?: string;
    device_discovery?: boolean;
};

const EMPTY_PROPERTIES: Record<string, FieldSchema> = {};
const EMPTY_GROUPS: Record<string, GroupOptions> = {};
const EMPTY_DEFINITIONS: Record<string, FieldSchema> = {};

const fieldLabel = (name: string, schema: FieldSchema) =>
    schema.title ?? name.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());

const asRecord = (value: unknown): Record<string, unknown> =>
    typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const resolveReference = (schema: FieldSchema, definitions: Record<string, FieldSchema>): FieldSchema => {
    if (schema.$ref === undefined) return schema;

    const definitionName = schema.$ref.replace('#/$defs/', '');
    const definition = definitions[definitionName];
    return definition === undefined ? schema : { ...definition, ...schema };
};

const updateObjectField = (value: unknown, name: string, fieldValue: unknown): Record<string, unknown> => ({
    ...asRecord(value),
    [name]: fieldValue,
});

const schemaDefault = (schema: FieldSchema, definitions: Record<string, FieldSchema>): unknown => {
    const resolvedSchema = resolveReference(schema, definitions);
    if (resolvedSchema.default !== undefined) return resolvedSchema.default;
    if (resolvedSchema.properties === undefined) return undefined;

    const defaults = schemaDefaults(resolvedSchema.properties, definitions);
    return Object.keys(defaults).length === 0 ? undefined : defaults;
};

const schemaDefaults = (
    properties: Record<string, FieldSchema>,
    definitions: Record<string, FieldSchema>
): Record<string, unknown> =>
    Object.fromEntries(
        Object.entries(properties).flatMap(([name, field]) => {
            const defaultValue = schemaDefault(field, definitions);
            return defaultValue === undefined ? [] : [[name, defaultValue]];
        })
    );

export const SchemaForm = ({ schema }: { schema: JsonSchema }) => {
    const { activeType, payload, setPayload, updatePayloadField } = useRobotForm();
    const [showDefaultFields, setShowDefaultFields] = useState(false);
    const properties = schema.properties ?? EMPTY_PROPERTIES;
    const groups = schema['x-physicalai-ui']?.groups ?? EMPTY_GROUPS;
    const definitions = schema.$defs ?? EMPTY_DEFINITIONS;
    const required = new Set(schema.required ?? []);

    useEffect(() => {
        if (Object.keys(payload).length !== 0) return;
        const defaults = schemaDefaults(properties, definitions);
        if (Object.keys(defaults).length !== 0) setPayload(defaults);
    }, [definitions, payload, properties, setPayload]);

    const renderFields = (
        fieldProperties: Record<string, FieldSchema>,
        fieldRequired: Set<string>,
        values: Record<string, unknown>,
        onChange: (name: string, value: unknown) => void,
        fieldGroups: Record<string, GroupOptions>
    ) => {
        const connectionGroups = Object.entries(fieldGroups).filter(([, group]) => group.device_discovery);
        const groupedFields = new Set(
            Object.entries(fieldProperties)
                .filter(([, field]) => connectionGroups.some(([name]) => field['x-physicalai-ui']?.group === name))
                .map(([name]) => name)
        );

        return (
            <>
                {connectionGroups.map(([name, options]) => (
                    <ConnectionField
                        key={name}
                        robotType={activeType!}
                        payload={values}
                        options={options}
                        onChange={onChange}
                    />
                ))}
                {Object.entries(fieldProperties)
                    .filter(([name]) => !groupedFields.has(name))
                    .map(([name, field]) => {
                        const resolvedField = resolveReference(field, definitions);
                        const isRequired =
                            fieldRequired.has(name) || resolvedField['x-physicalai-ui']?.required === true;
                        if (!isRequired && resolvedField.default !== undefined && !showDefaultFields)
                            return null;

                        if (resolvedField.properties !== undefined) {
                            return (
                                <View
                                    key={name}
                                    backgroundColor='gray-50'
                                    borderColor='gray-200'
                                    borderWidth='thin'
                                    padding='size-150'
                                >
                                    <Flex direction='column' gap='size-150'>
                                        <Heading level={4}>{fieldLabel(name, resolvedField)}</Heading>
                                        {renderFields(
                                            resolvedField.properties ?? EMPTY_PROPERTIES,
                                            new Set(resolvedField.required ?? []),
                                            asRecord(values[name]),
                                            (nestedName, nestedValue) =>
                                                onChange(
                                                    name,
                                                    updateObjectField(values[name], nestedName, nestedValue)
                                                ),
                                            (resolvedField['x-physicalai-ui'] as JsonSchema['x-physicalai-ui'])
                                                ?.groups ?? EMPTY_GROUPS
                                        )}
                                    </Flex>
                                </View>
                            );
                        }

                        if (resolvedField.type === 'object') return null;

                        return (
                            <SchemaField
                                key={name}
                                name={name}
                                schema={resolvedField}
                                value={values[name]}
                                isRequired={isRequired}
                                onChange={(value) => onChange(name, value)}
                            />
                        );
                    })}
            </>
        );
    };

    return (
        <Flex direction='column' gap='size-200'>
            <Switch isSelected={showDefaultFields} onChange={setShowDefaultFields}>
                Show default fields
            </Switch>
            {renderFields(properties, required, payload, updatePayloadField, groups)}
        </Flex>
    );
};
