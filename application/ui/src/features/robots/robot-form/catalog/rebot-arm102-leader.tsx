import { Flex, Item, Picker, Text, TextField } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../../api/client';
import type { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../../robot-types';
import { PermissionDeniedError } from '../../setup-wizard/so101/diagnostics-step-error';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot, RefreshRobotsButton, useIdentifyMutation } from './actions';

export interface ReBotArm102LeaderFormData {
    name: string;
    serial_number: string;
    connection_string: string;
    baudrate: string;
    unlock_on_connect: boolean;
    reset_multi_turn_on_connect: boolean;
    zero_on_connect: boolean;
}

export const getInitialReBotArm102LeaderFormData = (robot?: SchemaRobot): ReBotArm102LeaderFormData => ({
    name: robot?.name ?? '',
    serial_number: robot?.payload?.serial_number ?? '',
    connection_string: robot && 'connection_string' in robot.payload ? robot.payload.connection_string : '',
    baudrate: robot && 'baudrate' in robot.payload ? String(robot.payload.baudrate) : '1000000',
    unlock_on_connect: robot && 'unlock_on_connect' in robot.payload ? robot.payload.unlock_on_connect : true,
    reset_multi_turn_on_connect:
        robot && 'reset_multi_turn_on_connect' in robot.payload ? robot.payload.reset_multi_turn_on_connect : true,
    zero_on_connect: robot && 'zero_on_connect' in robot.payload ? robot.payload.zero_on_connect : false,
});

export const buildReBotArm102LeaderBody = (
    formData: ReBotArm102LeaderFormData,
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    if (!formData.serial_number) {
        return null;
    }

    const baudrate = Number.parseInt(formData.baudrate, 10);
    if (Number.isNaN(baudrate)) {
        return null;
    }

    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: {
            connection_string: formData.connection_string,
            serial_number: formData.serial_number,
            baudrate,
            unlock_on_connect: formData.unlock_on_connect,
            reset_multi_turn_on_connect: formData.reset_multi_turn_on_connect,
            zero_on_connect: formData.zero_on_connect,
        },
    } as SchemaRobotInput;
};

const getDeviceKey = ({ serial_number, connection_string }: { serial_number: string; connection_string: string }) => {
    if (serial_number !== '') {
        return `serial:${serial_number}`;
    }
    return `port:${connection_string}`;
};

const normalizeSerialNumber = (serialNumber: string | null | undefined): string => {
    if (!serialNumber || serialNumber === 'no_serial') {
        return '';
    }
    return serialNumber;
};

export const ReBotArm102LeaderFormFields = () => {
    const serialDevicesQuery = $api.useSuspenseQuery('get', '/api/hardware/serial_devices');
    const { formData, updateField, activeType } = useRobotFormFields<ReBotArm102LeaderFormData>();

    const identifyMutation = useIdentifyMutation();
    const identifyRobot = buildReBotArm102LeaderBody(formData, activeType, uuidv4());

    const selectedKey =
        formData.serial_number !== '' || formData.connection_string !== ''
            ? getDeviceKey({
                  serial_number: normalizeSerialNumber(formData.serial_number),
                  connection_string: formData.connection_string,
              })
            : null;

    return (
        <>
            <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
                <Picker
                    name='payload.device_key'
                    label='Select robot'
                    isRequired
                    width='100%'
                    selectedKey={selectedKey}
                    onSelectionChange={(selectedKey) => {
                        const device = serialDevicesQuery.data.find((d) => getDeviceKey(d) === selectedKey);

                        if (device === undefined) {
                            return;
                        }

                        const serial_number = normalizeSerialNumber(device.serial_number);

                        updateField('serial_number', serial_number);
                        updateField('connection_string', device?.connection_string ?? '');
                    }}
                >
                    {serialDevicesQuery.data.map((serial_device) => {
                        const serial_number = normalizeSerialNumber(serial_device.serial_number);
                        const hasSerial = serial_number !== '';
                        const label = hasSerial ? serial_number : 'No serial number';

                        return (
                            <Item
                                key={getDeviceKey({
                                    serial_number,
                                    connection_string: serial_device.connection_string,
                                })}
                                textValue={label}
                            >
                                <Text>{label}</Text>
                                <Text slot='description'>{serial_device.connection_string}</Text>
                            </Item>
                        );
                    })}
                </Picker>

                <Flex gap='size-100'>
                    <RefreshRobotsButton />
                    <IdentifyRobot identifyMutation={identifyMutation} robot={identifyRobot} />
                </Flex>
            </Flex>

            <TextField
                isRequired
                label='Baudrate'
                width='100%'
                value={formData.baudrate}
                onChange={(baudrate) => {
                    updateField('baudrate', baudrate);
                }}
                placeholder='1000000'
            />

            <Flex gap='size-100'>
                <Picker
                    label='Unlock on connect'
                    isRequired
                    width='33%'
                    selectedKey={formData.unlock_on_connect ? 'true' : 'false'}
                    onSelectionChange={(selected) => {
                        updateField('unlock_on_connect', selected === 'true');
                    }}
                >
                    <Item key={'true'}>true</Item>
                    <Item key={'false'}>false</Item>
                </Picker>
                <Picker
                    label='Reset multi-turn'
                    isRequired
                    width='33%'
                    selectedKey={formData.reset_multi_turn_on_connect ? 'true' : 'false'}
                    onSelectionChange={(selected) => {
                        updateField('reset_multi_turn_on_connect', selected === 'true');
                    }}
                >
                    <Item key={'true'}>true</Item>
                    <Item key={'false'}>false</Item>
                </Picker>
                <Picker
                    label='Zero on connect'
                    isRequired
                    width='33%'
                    selectedKey={formData.zero_on_connect ? 'true' : 'false'}
                    onSelectionChange={(selected) => {
                        updateField('zero_on_connect', selected === 'true');
                    }}
                >
                    <Item key={'true'}>true</Item>
                    <Item key={'false'}>false</Item>
                </Picker>
            </Flex>

            {identifyMutation.isError && <PermissionDeniedError port={formData.connection_string} />}
        </>
    );
};
