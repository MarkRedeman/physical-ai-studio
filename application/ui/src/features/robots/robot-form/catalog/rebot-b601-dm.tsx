import { Flex, Item, Picker, Text, TextField } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../../api/client';
import type { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../../robot-types';
import { PermissionDeniedError } from '../../setup-wizard/so101/diagnostics-step-error';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot, RefreshRobotsButton, useIdentifyMutation } from './actions';

export interface ReBotB601DMFormData {
    name: string;
    serial_number: string;
    connection_string: string;
    can_adapter: 'damiao' | 'socketcan';
    dm_serial_baud: string;
    disable_torque_on_disconnect: boolean;
    force_pos_torque_ratio: string;
}

export const getInitialReBotB601DMFormData = (robot?: SchemaRobot): ReBotB601DMFormData => ({
    name: robot?.name ?? '',
    serial_number: robot?.payload?.serial_number ?? '',
    connection_string: robot && 'connection_string' in robot.payload ? robot.payload.connection_string : '',
    can_adapter:
        robot && 'can_adapter' in robot.payload && robot.payload.can_adapter === 'socketcan' ? 'socketcan' : 'damiao',
    dm_serial_baud: robot && 'dm_serial_baud' in robot.payload ? String(robot.payload.dm_serial_baud) : '921600',
    disable_torque_on_disconnect:
        robot && 'disable_torque_on_disconnect' in robot.payload ? robot.payload.disable_torque_on_disconnect : true,
    force_pos_torque_ratio:
        robot && 'force_pos_torque_ratio' in robot.payload ? String(robot.payload.force_pos_torque_ratio) : '0.1',
});

export const buildReBotB601DMBody = (
    formData: ReBotB601DMFormData,
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    if (!formData.serial_number) {
        return null;
    }

    const dmSerialBaud = Number.parseInt(formData.dm_serial_baud, 10);
    const forcePosTorqueRatio = Number.parseFloat(formData.force_pos_torque_ratio);

    if (Number.isNaN(dmSerialBaud) || Number.isNaN(forcePosTorqueRatio)) {
        return null;
    }

    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: {
            connection_string: formData.connection_string,
            serial_number: formData.serial_number,
            can_adapter: formData.can_adapter,
            dm_serial_baud: dmSerialBaud,
            disable_torque_on_disconnect: formData.disable_torque_on_disconnect,
            force_pos_torque_ratio: forcePosTorqueRatio,
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

export const ReBotB601DMFormFields = () => {
    const serialDevicesQuery = $api.useSuspenseQuery('get', '/api/hardware/serial_devices');
    const { formData, updateField, activeType } = useRobotFormFields<ReBotB601DMFormData>();

    const identifyMutation = useIdentifyMutation();
    const identifyRobot = buildReBotB601DMBody(formData, activeType, uuidv4());

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

                        updateField('serial_number', String(serial_number));
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

            <Flex gap='size-100'>
                <Picker
                    label='CAN adapter'
                    isRequired
                    width='50%'
                    selectedKey={formData.can_adapter}
                    onSelectionChange={(selected) => {
                        updateField('can_adapter', selected === 'socketcan' ? 'socketcan' : 'damiao');
                    }}
                >
                    <Item key={'damiao'}>damiao</Item>
                    <Item key={'socketcan'}>socketcan</Item>
                </Picker>
                <TextField
                    isRequired
                    label='DM serial baud'
                    width='50%'
                    value={formData.dm_serial_baud}
                    onChange={(dm_serial_baud) => {
                        updateField('dm_serial_baud', dm_serial_baud);
                    }}
                    placeholder='921600'
                />
            </Flex>

            <Flex gap='size-100'>
                <TextField
                    isRequired
                    label='Force-pos torque ratio'
                    width='50%'
                    value={formData.force_pos_torque_ratio}
                    onChange={(force_pos_torque_ratio) => {
                        updateField('force_pos_torque_ratio', force_pos_torque_ratio);
                    }}
                    placeholder='0.1'
                />
                <Picker
                    label='Disable torque on disconnect'
                    isRequired
                    width='50%'
                    selectedKey={formData.disable_torque_on_disconnect ? 'true' : 'false'}
                    onSelectionChange={(selected) => {
                        updateField('disable_torque_on_disconnect', selected === 'true');
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
