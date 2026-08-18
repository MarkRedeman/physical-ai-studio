import { useEffect, useState } from 'react';

import { Flex, Heading, Switch, Text, View } from '@geti-ui/ui';

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
import { ControlOptions, FieldSchema, JsonSchema, RobotUiInfo, SectionOptions } from './types';

const EMPTY_SECTIONS: SectionOptions[] = [];
const EMPTY_INFOS: RobotUiInfo[] = [];

const boundFieldNamesForSections = (sections: SectionOptions[]): Set<string> =>
    new Set(
        sections.flatMap((section) =>
            (section.controls ?? []).flatMap((control) =>
                control.kind === 'connection'
                    ? [control.bind.connection, ...(control.bind.serial_number === undefined ? [] : [control.bind.serial_number])]
                    : []
            )
        )
    );

const isFieldVisible = (
    field: FieldSchema,
    fieldName: string,
    fieldRequired: Set<string>,
    definitions: Record<string, FieldSchema>,
    showDefaultFields: boolean
) => {
    const resolvedField = resolveReference(field, definitions);
    const isRequired = fieldRequired.has(fieldName) || resolvedField['x-physicalai-ui']?.required === true;
    if (!isRequired && resolvedField.default !== undefined && !showDefaultFields) return false;
    if (resolvedField.type === 'object' && resolvedField.properties === undefined) return false;
    return true;
};

export const SchemaForm = ({ schema }: { schema: JsonSchema }) => {
    const { activeType, payload, setPayload, updatePayloadField } = useRobotForm();
    const [showDefaultFields, setShowDefaultFields] = useState(false);
    const properties = schema.properties ?? EMPTY_PROPERTIES;
    const sections = schema['x-physicalai-ui']?.sections ?? EMPTY_SECTIONS;
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

    const renderFieldEntries = (
        entries: [string, FieldSchema][],
        fieldRequired: Set<string>,
        values: Record<string, unknown>,
        onChange: (name: string, value: unknown) => void,
        nestedSections: SectionOptions[],
        sectionFieldSource: Record<string, FieldSchema>
    ) => {
        const renderControls = (controls: ControlOptions[]) =>
            controls.map((control, index) => {
                if (control.kind === 'connection') {
                    return (
                        <Flex key={`connection-${control.label ?? index}`} direction='column' gap='size-100'>
                            {renderInfos(control.infos ?? EMPTY_INFOS)}
                            <ConnectionField robotType={activeType!} payload={values} options={control} onChange={onChange} />
                        </Flex>
                    );
                }
                return null;
            });

        return (
            <>
                {nestedSections.map((section) => {
                    const sectionEntries =
                        section.fields === undefined
                            ? []
                            : section.fields
                                  .map((name) => {
                                      const field = sectionFieldSource[name];
                                      return field === undefined ? null : ([name, field] as [string, FieldSchema]);
                                  })
                                  .filter((entry): entry is [string, FieldSchema] => entry !== null);
                    const sectionBoundFieldNames = boundFieldNamesForSections([section]);
                    const visibleSectionEntries = sectionEntries.filter(
                        ([name]) => !sectionBoundFieldNames.has(name)
                    );
                    const renderableSectionEntries = visibleSectionEntries.filter(([name, field]) =>
                        isFieldVisible(field, name, fieldRequired, definitions, showDefaultFields)
                    );
                    if (
                        renderableSectionEntries.length === 0 &&
                        (section.controls ?? []).length === 0 &&
                        (section.infos ?? []).length === 0
                    )
                        return null;

                    return (
                        <Flex key={section.id} direction='column' gap='size-150'>
                            {section.title !== undefined && <Heading level={4}>{section.title}</Heading>}
                            {section.description !== undefined && <Text>{section.description}</Text>}
                            {renderInfos(section.infos ?? EMPTY_INFOS)}
                            {renderControls(section.controls ?? [])}
                            {renderFieldEntries(
                                renderableSectionEntries,
                                fieldRequired,
                                values,
                                onChange,
                                EMPTY_SECTIONS,
                                sectionFieldSource
                            )}
                        </Flex>
                    );
                })}
                {entries.map(([name, field]) => {
                    const resolvedField = resolveReference(field, definitions);
                    const isRequired = fieldRequired.has(name) || resolvedField['x-physicalai-ui']?.required === true;
                    if (!isRequired && resolvedField.default !== undefined && !showDefaultFields) return null;

                    if (resolvedField.properties !== undefined) {
                        const propertyEntries = Object.entries(resolvedField.properties ?? EMPTY_PROPERTIES);
                        const nestedUi = (resolvedField['x-physicalai-ui'] as JsonSchema['x-physicalai-ui']) ?? {};
                        const nestedSectionsFromUi = nestedUi.sections ?? EMPTY_SECTIONS;
                        const nestedSectionFieldNames = new Set(
                            nestedSectionsFromUi.flatMap((section) => section.fields ?? [])
                        );
                        const nestedBoundFieldNames = boundFieldNamesForSections(nestedSectionsFromUi);
                        const nestedDefaultEntries = propertyEntries.filter(
                            ([nestedName]) => !nestedSectionFieldNames.has(nestedName) && !nestedBoundFieldNames.has(nestedName)
                        );

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
                                    {renderInfos(nestedUi.infos ?? EMPTY_INFOS)}
                                    {renderFieldEntries(
                                        nestedDefaultEntries,
                                        new Set(resolvedField.required ?? []),
                                        asRecord(values[name]),
                                        (nestedName, nestedValue) =>
                                            onChange(name, updateObjectField(values[name], nestedName, nestedValue)),
                                        nestedSectionsFromUi,
                                        resolvedField.properties ?? EMPTY_PROPERTIES
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

    const sectionFieldNames = new Set(sections.flatMap((section) => section.fields ?? []));
    const boundFieldNames = boundFieldNamesForSections(sections);
    const defaultEntries = Object.entries(properties).filter(
        ([name]) => !sectionFieldNames.has(name) && !boundFieldNames.has(name)
    );

    return (
        <Flex direction='column' gap='size-200'>
            {renderInfos(infos)}
            <Switch isSelected={showDefaultFields} onChange={setShowDefaultFields}>
                Show default fields
            </Switch>
            {renderFieldEntries(defaultEntries, required, payload, updatePayloadField, sections, properties)}
        </Flex>
    );
};
