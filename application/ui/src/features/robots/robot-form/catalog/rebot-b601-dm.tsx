import { Flex, Item, Picker, Text, TextField } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { useRobotForm, useSetRobotForm } from '../provider';
import { RefreshRobotsButton } from './actions';

export const ReBotB601DMFormFields = () => {
    const serialDevicesQuery = $api.useSuspenseQuery('get', '/api/hardware/serial_devices');
    const robotForm = useRobotForm();
    const setRobotForm = useSetRobotForm();

    return (
        <Flex direction='column' gap='size-100'>
            <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
                <Picker
                    name='payload.serial_number'
                    label='Select robot'
                    isRequired
                    width='100%'
                    selectedKey={robotForm.serial_number}
                    onSelectionChange={(serial_number) => {
                        const device = serialDevicesQuery.data.find((d) => d.serial_number === serial_number);

                        setRobotForm((oldForm) => ({
                            ...oldForm,
                            serial_number: String(serial_number),
                            connection_string: device?.connection_string ?? '',
                        }));
                    }}
                >
                    {serialDevicesQuery.data.map((serial_device) => {
                        return (
                            <Item key={serial_device.serial_number} textValue={serial_device.serial_number}>
                                <Text>{serial_device.serial_number}</Text>
                                <Text slot='description'>{serial_device.connection_string}</Text>
                            </Item>
                        );
                    })}
                </Picker>
                <RefreshRobotsButton />
            </Flex>

            <Flex gap='size-100'>
                <Picker
                    name='payload.can_adapter'
                    label='CAN adapter'
                    isRequired
                    width='50%'
                    selectedKey={robotForm.can_adapter}
                    onSelectionChange={(selected) => {
                        const can_adapter = selected === 'socketcan' ? 'socketcan' : 'damiao';
                        setRobotForm((oldForm) => ({ ...oldForm, can_adapter }));
                    }}
                >
                    <Item key={'damiao'}>damiao</Item>
                    <Item key={'socketcan'}>socketcan</Item>
                </Picker>
                <TextField
                    name='payload.dm_serial_baud'
                    isRequired
                    label='DM serial baud'
                    width='50%'
                    value={robotForm.dm_serial_baud}
                    onChange={(dm_serial_baud) => {
                        setRobotForm((oldForm) => ({ ...oldForm, dm_serial_baud }));
                    }}
                    placeholder='921600'
                />
            </Flex>

            <Flex gap='size-100'>
                <TextField
                    name='payload.force_pos_torque_ratio'
                    isRequired
                    label='Force-pos torque ratio'
                    width='50%'
                    value={robotForm.force_pos_torque_ratio}
                    onChange={(force_pos_torque_ratio) => {
                        setRobotForm((oldForm) => ({ ...oldForm, force_pos_torque_ratio }));
                    }}
                    placeholder='0.1'
                />
                <Picker
                    name='payload.disable_torque_on_disconnect'
                    label='Disable torque on disconnect'
                    isRequired
                    width='50%'
                    selectedKey={robotForm.disable_torque_on_disconnect ? 'true' : 'false'}
                    onSelectionChange={(selected) => {
                        setRobotForm((oldForm) => ({
                            ...oldForm,
                            disable_torque_on_disconnect: selected === 'true',
                        }));
                    }}
                >
                    <Item key={'true'}>true</Item>
                    <Item key={'false'}>false</Item>
                </Picker>
            </Flex>
        </Flex>
    );
};
