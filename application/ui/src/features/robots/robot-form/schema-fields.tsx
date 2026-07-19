import { Item, Picker, Switch, TextField } from '@geti-ui/ui';

export type FieldSchema = {
    type?: string;
    title?: string;
    description?: string;
    default?: unknown;
    enum?: unknown[];
    $ref?: string;
    properties?: Record<string, FieldSchema>;
    additionalProperties?: FieldSchema | boolean;
    required?: string[];
    ['x-physicalai-ui']?: FieldOptions;
};

export type FieldOptions = {
    group?: string;
    widget?: 'device-selector';
    required?: boolean;
    groups?: Record<
        string,
        {
            title?: string;
            device_discovery?: boolean;
            identify?: boolean;
            connection_key?: string;
            serial_number_key?: string;
            manual_entry?: boolean;
        }
    >;
};

type FieldProps = {
    name: string;
    schema: FieldSchema;
    value: unknown;
    isRequired: boolean;
    onChange: (value: unknown) => void;
};

const fieldLabel = (name: string, schema: FieldSchema) =>
    schema.title ?? name.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());

const commonProps = ({ name, schema, isRequired }: Pick<FieldProps, 'name' | 'schema' | 'isRequired'>) => ({
    label: fieldLabel(name, schema),
    description: schema.description,
    isRequired,
    width: '100%' as const,
});

export const EnumPickerField = ({ name, schema, value, isRequired, onChange }: FieldProps) => (
    <Picker
        {...commonProps({ name, schema, isRequired })}
        selectedKey={String(value ?? '')}
        onSelectionChange={onChange}
    >
        {(schema.enum ?? []).map((option) => (
            <Item key={String(option)}>{String(option)}</Item>
        ))}
    </Picker>
);

export const BooleanField = ({ name, schema, value, isRequired, onChange }: FieldProps) => (
    <Switch isRequired={isRequired} isSelected={Boolean(value)} onChange={onChange}>
        {fieldLabel(name, schema)}
    </Switch>
);

export const TextFieldValue = ({ name, schema, value, isRequired, onChange }: FieldProps) => {
    const isNumeric = schema.type === 'integer' || schema.type === 'number';
    return (
        <TextField
            {...commonProps({ name, schema, isRequired })}
            type={isNumeric ? 'number' : 'text'}
            value={value === undefined || value === null ? '' : String(value)}
            onChange={(next) =>
                onChange(
                    schema.type === 'integer'
                        ? Number.parseInt(next, 10)
                        : schema.type === 'number'
                          ? Number.parseFloat(next)
                          : next
                )
            }
        />
    );
};

export const SchemaField = (props: FieldProps) => {
    if (props.schema.enum) return <EnumPickerField {...props} />;
    if (props.schema.type === 'boolean') return <BooleanField {...props} />;
    return <TextFieldValue {...props} />;
};
