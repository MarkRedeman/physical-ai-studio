import { Flex, Item, Picker, Text, TextField } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { useRobotForm, useSetRobotForm } from '../provider';
import { RefreshRobotsButton } from './actions';

export const ReBotArm102LeaderFormFields = () => {
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

            <TextField
                name='payload.baudrate'
                isRequired
                label='Baudrate'
                width='100%'
                value={robotForm.baudrate}
                onChange={(baudrate) => {
                    setRobotForm((oldForm) => ({ ...oldForm, baudrate }));
                }}
                placeholder='1000000'
            />

            <Flex gap='size-100'>
                <Picker
                    name='payload.unlock_on_connect'
                    label='Unlock on connect'
                    isRequired
                    width='33%'
                    selectedKey={robotForm.unlock_on_connect ? 'true' : 'false'}
                    onSelectionChange={(selected) => {
                        setRobotForm((oldForm) => ({ ...oldForm, unlock_on_connect: selected === 'true' }));
                    }}
                >
                    <Item key={'true'}>true</Item>
                    <Item key={'false'}>false</Item>
                </Picker>
                <Picker
                    name='payload.reset_multi_turn_on_connect'
                    label='Reset multi-turn'
                    isRequired
                    width='33%'
                    selectedKey={robotForm.reset_multi_turn_on_connect ? 'true' : 'false'}
                    onSelectionChange={(selected) => {
                        setRobotForm((oldForm) => ({
                            ...oldForm,
                            reset_multi_turn_on_connect: selected === 'true',
                        }));
                    }}
                >
                    <Item key={'true'}>true</Item>
                    <Item key={'false'}>false</Item>
                </Picker>
                <Picker
                    name='payload.zero_on_connect'
                    label='Zero on connect'
                    isRequired
                    width='33%'
                    selectedKey={robotForm.zero_on_connect ? 'true' : 'false'}
                    onSelectionChange={(selected) => {
                        setRobotForm((oldForm) => ({ ...oldForm, zero_on_connect: selected === 'true' }));
                    }}
                >
                    <Item key={'true'}>true</Item>
                    <Item key={'false'}>false</Item>
                </Picker>
            </Flex>
        </Flex>
    );
};
