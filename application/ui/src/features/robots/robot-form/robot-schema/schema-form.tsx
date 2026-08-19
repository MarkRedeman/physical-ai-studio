import { useEffect, useState } from 'react';

import { Flex, Heading, Switch, Text, View } from '@geti-ui/ui';

import { useRobotForm } from '../provider';
import { ConnectionField } from './first-party-fields/connection-field';
import { InfoField } from './first-party-fields/info-field';
import { SchemaField } from './schema-field';
import {
    asRecord,
    EMPTY_DEFINITIONS,
    EMPTY_PROPERTIES,
    fieldLabel,
    resolveReference,
    schemaDefaults,
    updateObjectField,
} from './schema-utils';
import { FieldSchema, JsonSchema, ModelUiOptions, RobotUiItem } from './types';

const EMPTY_ITEMS: RobotUiItem[] = [];

const isUiItems = (value: unknown): value is ModelUiOptions => Array.isArray(value);

const fieldNamesOwnedByItems = (items: RobotUiItem[]): Set<string> =>
    new Set(
        items.flatMap((item) => {
            if (item.kind === 'field') return [item.name];
            if (item.kind === 'connection') {
                return [
                    item.bind.connection,
                    ...(item.bind.serial_number === undefined ? [] : [item.bind.serial_number]),
                ];
            }
            if (item.kind === 'section') return [...fieldNamesOwnedByItems(item.items)];
            return [];
        })
    );

export const SchemaForm = ({ schema }: { schema: JsonSchema }) => {
    const { activeType, payload, setPayload, updatePayloadField } = useRobotForm();
    const [showDefaultFields, setShowDefaultFields] = useState(false);
    const properties = schema.properties ?? EMPTY_PROPERTIES;
    const definitions = schema.$defs ?? EMPTY_DEFINITIONS;
    const required = new Set(schema.required ?? []);
    const items = schema['x-physicalai-ui'] ?? EMPTY_ITEMS;

    useEffect(() => {
        if (Object.keys(payload).length !== 0) return;
        const defaults = schemaDefaults(properties, definitions);
        if (Object.keys(defaults).length !== 0) setPayload(defaults);
    }, [definitions, payload, properties, setPayload]);

    const isFieldVisible = (field: FieldSchema, fieldName: string, fieldRequired: Set<string>) => {
        const resolvedField = resolveReference(field, definitions);
        const fieldUi = resolvedField['x-physicalai-ui'];
        const isRequired = fieldRequired.has(fieldName) || (!isUiItems(fieldUi) && fieldUi?.required === true);
        if (!isRequired && resolvedField.default !== undefined && !showDefaultFields) return false;
        return resolvedField.type !== 'object' || resolvedField.properties !== undefined;
    };

    const renderField = (
        name: string,
        field: FieldSchema,
        fieldRequired: Set<string>,
        values: Record<string, unknown>,
        onChange: (name: string, value: unknown) => void
    ) => {
        if (!isFieldVisible(field, name, fieldRequired)) return null;

        const resolvedField = resolveReference(field, definitions);
        const fieldUi = resolvedField['x-physicalai-ui'];
        const isRequired = fieldRequired.has(name) || (!isUiItems(fieldUi) && fieldUi?.required === true);
        if (resolvedField.properties !== undefined) {
            const nestedItems = isUiItems(fieldUi) ? fieldUi : EMPTY_ITEMS;
            const nestedProperties = resolvedField.properties ?? EMPTY_PROPERTIES;
            return (
                <View key={name} backgroundColor='gray-50' borderColor='gray-200' borderWidth='thin' padding='size-150'>
                    <Flex direction='column' gap='size-150'>
                        <Heading level={4}>{fieldLabel(name, resolvedField)}</Heading>
                        {renderItems(
                            nestedItems,
                            nestedProperties,
                            new Set(resolvedField.required ?? []),
                            asRecord(values[name]),
                            (nestedName, nestedValue) =>
                                onChange(name, updateObjectField(values[name], nestedName, nestedValue)),
                            true
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
    };

    const itemIsRenderable = (
        item: RobotUiItem,
        itemProperties: Record<string, FieldSchema>,
        itemRequired: Set<string>
    ): boolean => {
        if (item.kind === 'info' || item.kind === 'connection') return true;
        if (item.kind === 'field') {
            const field = itemProperties[item.name];
            return field !== undefined && isFieldVisible(field, item.name, itemRequired);
        }
        return item.items.some((child) => itemIsRenderable(child, itemProperties, itemRequired));
    };

    const renderItems = (
        itemList: RobotUiItem[],
        itemProperties: Record<string, FieldSchema>,
        itemRequired: Set<string>,
        values: Record<string, unknown>,
        onChange: (name: string, value: unknown) => void,
        renderUnownedFields: boolean
    ) => {
        const ownedFields = fieldNamesOwnedByItems(itemList);
        const unsectionedFields = Object.entries(itemProperties).filter(([name]) => !ownedFields.has(name));

        return (
            <>
                {itemList.map((item, index) => {
                    if (item.kind === 'info') return <InfoField key={`info-${index}`} info={item} />;
                    if (item.kind === 'connection') {
                        return (
                            <ConnectionField
                                key={`connection-${index}`}
                                robotType={activeType!}
                                payload={values}
                                options={item}
                                onChange={onChange}
                            />
                        );
                    }
                    if (item.kind === 'field') {
                        const field = itemProperties[item.name];
                        return field === undefined
                            ? null
                            : renderField(item.name, field, itemRequired, values, onChange);
                    }
                    if (!itemIsRenderable(item, itemProperties, itemRequired)) return null;
                    return (
                        <Flex key={item.id} direction='column' gap='size-150'>
                            {item.title !== undefined && <Heading level={4}>{item.title}</Heading>}
                            {item.description !== undefined && <Text>{item.description}</Text>}
                            {renderItems(item.items, itemProperties, itemRequired, values, onChange, false)}
                        </Flex>
                    );
                })}
                {renderUnownedFields &&
                    unsectionedFields.map(([name, field]) => renderField(name, field, itemRequired, values, onChange))}
            </>
        );
    };

    return (
        <Flex direction='column' gap='size-200'>
            <Switch isSelected={showDefaultFields} onChange={setShowDefaultFields}>
                Show default fields
            </Switch>
            {renderItems(items, properties, required, payload, updatePayloadField, true)}
        </Flex>
    );
};
