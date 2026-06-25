import { Flex, Item, Picker, Text } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { PermissionDeniedError } from '../../setup-wizard/so101/diagnostics-step-error';
import { useRobotForm, useSetRobotForm } from '../provider';
import { IdentifyRobot, RefreshRobotsButton, useIdentifyMutation } from './actions';

export const SO101FormFields = () => {
    const serialDevicesQuery = $api.useSuspenseQuery('get', '/api/hardware/serial_devices');

    const robotForm = useRobotForm();
    const setRobotForm = useSetRobotForm();

    const identifyMutation = useIdentifyMutation();

    return (
        <>
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

                <Flex gap='size-100'>
                    <RefreshRobotsButton />
                    <IdentifyRobot identifyMutation={identifyMutation} robotForm={robotForm} />
                </Flex>
            </Flex>

            {identifyMutation.isError && <PermissionDeniedError port={robotForm.connection_string} />}
        </>
    );
};
