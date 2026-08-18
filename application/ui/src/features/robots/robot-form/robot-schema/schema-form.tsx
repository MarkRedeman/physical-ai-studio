import { useEffect, useState } from 'react';

import { Flex, Heading, Switch, View } from '@geti-ui/ui';

import { useRobotForm } from '../provider';
import { ConnectionField } from './first-party-fields/connection-field';
import { InfoField } from './first-party-fields/info-field';
import { SchemaField } from './schema-field';
import {
    EMPTY_DEFINITIONS,
    EMPTY_PROPERTIES,
    asRecord,
    fieldLabel,
    resolveReference,
    schemaDefaults,
    updateObjectField,
} from './schema-utils';
import { FieldSchema, GroupOptions, JsonSchema, RobotUiInfo } from './types';

const EMPTY_GROUPS: Record<string, GroupOptions> = {};
const EMPTY_INFOS: RobotUiInfo[] = [];

export const SchemaForm = ({ schema }: { schema: JsonSchema }) => {
    const { activeType, payload, setPayload, updatePayloadField } = useRobotForm();
    const [showDefaultFields, setShowDefaultFields] = useState(false);
    const properties = schema.properties ?? EMPTY_PROPERTIES;
    const groups = schema['x-physicalai-ui']?.groups ?? EMPTY_GROUPS;
    const definitions = schema.$defs ?? EMPTY_DEFINITIONS;
    const required = new Set(schema.required ?? []);
    const infos = schema['x-physicalai-ui']?.infos ?? EMPTY_INFOS;

    useEffect(() => {
        if (Object.keys(payload).length !== 0) return;
        const defaults = schemaDefaults(properties, definitions);
        if (Object.keys(defaults).length !== 0) setPayload(defaults);
    }, [definitions, payload, properties, setPayload]);

    const renderInfos = (fieldInfos: RobotUiInfo[]) =>
        fieldInfos.map((info, index) => <InfoField key={`${info.title ?? info.text}-${index}`} info={info} />);

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
                    <Flex key={name} direction='column' gap='size-100'>
                        {renderInfos(options.infos ?? EMPTY_INFOS)}
                        <ConnectionField robotType={activeType!} payload={values} options={options} onChange={onChange} />
                    </Flex>
                ))}
                {Object.entries(fieldProperties)
                    .filter(([name]) => !groupedFields.has(name))
                    .map(([name, field]) => {
                        const resolvedField = resolveReference(field, definitions);
                        const isRequired =
                            fieldRequired.has(name) || resolvedField['x-physicalai-ui']?.required === true;
                        if (!isRequired && resolvedField.default !== undefined && !showDefaultFields) return null;

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
                                        {renderInfos((resolvedField['x-physicalai-ui'] as JsonSchema['x-physicalai-ui'])?.infos ?? EMPTY_INFOS)}
                                        {renderFields(
                                            resolvedField.properties ?? EMPTY_PROPERTIES,
                                            new Set(resolvedField.required ?? []),
                                            asRecord(values[name]),
                                            (nestedName, nestedValue) =>
                                                onChange(name, updateObjectField(values[name], nestedName, nestedValue)),
                                            (resolvedField['x-physicalai-ui'] as JsonSchema['x-physicalai-ui'])?.groups ??
                                                EMPTY_GROUPS
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
            {renderInfos(infos)}
            <Switch isSelected={showDefaultFields} onChange={setShowDefaultFields}>
                Show default fields
            </Switch>
            {renderFields(properties, required, payload, updatePayloadField, groups)}
        </Flex>
    );
};
