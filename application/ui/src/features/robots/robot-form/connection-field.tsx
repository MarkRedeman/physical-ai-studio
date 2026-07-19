import { ActionButton, ComboBox, Flex, Icon, Item, Text } from '@geti-ui/ui';
import { Refresh } from '@geti-ui/ui/icons';

import { useCatalogIdentifyMutation, useDiscoverRobotsQuery } from '../robot-catalog.hooks';

export type ConnectionGroupOptions = {
    title?: string;
    identify?: boolean;
    connection_key?: string;
    serial_number_key?: string;
    manual_entry?: boolean;
};

type Device = { serial_number: string | null; connection_string: string | null };

type ComboBoxFieldProps = {
    label: string;
    value: string;
    devices: Device[];
    allowsCustomValue: boolean;
    onInputChange: (value: string) => void;
    onSelectionChange: (key: string | number | null) => void;
};

const normalizedSerialNumber = (serialNumber: string | null) =>
    serialNumber === 'no_serial' ? '' : (serialNumber ?? '');

const deviceKey = (device: Device) => {
    const serialNumber = normalizedSerialNumber(device.serial_number);
    return serialNumber ? `serial:${serialNumber}` : `port:${device.connection_string ?? ''}`;
};

const deviceTextValue = (device: Device) =>
    normalizedSerialNumber(device.serial_number) || device.connection_string || '';

const ComboBoxField = ({
    label,
    value,
    devices,
    allowsCustomValue,
    onInputChange,
    onSelectionChange,
}: ComboBoxFieldProps) => (
    <ComboBox
        label={label}
        width='100%'
        allowsCustomValue={allowsCustomValue}
        inputValue={value}
        onInputChange={onInputChange}
        onSelectionChange={onSelectionChange}
    >
        {devices.map((device) => (
            <Item key={deviceKey(device)} textValue={deviceTextValue(device)}>
                <Text>{device.serial_number ?? 'No serial number'}</Text>
                <Text slot='description'>{device.connection_string ?? ''}</Text>
            </Item>
        ))}
    </ComboBox>
);

type ConnectionFieldProps = {
    robotType: string;
    payload: Record<string, unknown>;
    options: ConnectionGroupOptions;
    onChange: (field: string, value: unknown) => void;
};

export const ConnectionField = ({ robotType, payload, options, onChange }: ConnectionFieldProps) => {
    const discover = useDiscoverRobotsQuery(robotType);
    const identify = useCatalogIdentifyMutation();
    const connectionKey = options.connection_key;
    const serialNumberKey = options.serial_number_key;
    const value = connectionKey === undefined ? '' : String(payload[connectionKey] ?? '');

    const setManualValue = (next: string) => {
        if ((discover.data ?? []).some((device) => deviceTextValue(device) === next)) return;
        if (connectionKey === undefined) return;
        onChange(connectionKey, next);
        if (serialNumberKey !== undefined) onChange(serialNumberKey, '');
    };

    return (
        <Flex direction='column' gap='size-100'>
            <Flex gap='size-100' alignItems='end'>
                <ComboBoxField
                    label={options.title ?? 'Connection'}
                    value={value}
                    devices={discover.data ?? []}
                    allowsCustomValue={options.manual_entry !== false}
                    onInputChange={setManualValue}
                    onSelectionChange={(key) => {
                        const device = (discover.data ?? []).find((item) => deviceKey(item) === key);
                        if (device === undefined || connectionKey === undefined) return;
                        onChange(connectionKey, device.connection_string ?? '');
                        if (serialNumberKey !== undefined)
                            onChange(serialNumberKey, normalizedSerialNumber(device.serial_number));
                    }}
                />
                <ActionButton onPress={() => discover.refetch()} isDisabled={discover.isFetching}>
                    <Icon>
                        <Refresh />
                    </Icon>
                </ActionButton>
                {options.identify && (
                    <ActionButton
                        onPress={() => identify.mutate({ params: { path: { robot_type: robotType } }, body: payload })}
                        isDisabled={identify.isPending}
                    >
                        Identify
                    </ActionButton>
                )}
            </Flex>
        </Flex>
    );
};
