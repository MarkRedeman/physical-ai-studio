import { Flex, TextField, View } from '@geti-ui/ui';

import { useRobotForm, useSetRobotForm } from '../provider';
import { IdentifyRobot, useIdentifyMutation } from './actions';

export const BiManualWidowxAIFormFields = () => {
    const robotForm = useRobotForm();
    const setRobotForm = useSetRobotForm();

    const identifyMutation = useIdentifyMutation();

    return (
        <>
            <Flex direction='column' gap='size-100' width='100%'>
                <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
                    <TextField
                        name='payload.connection_string_left'
                        isRequired
                        label='Left arm IP address'
                        width='100%'
                        value={robotForm.connection_string_left ?? ''}
                        onChange={(connection_string_left) => {
                            setRobotForm((oldForm) => ({
                                ...oldForm,
                                connection_string_left,
                                serial_number: '',
                            }));
                        }}
                        placeholder='192.168.1.2'
                    />
                    <View>
                        <IdentifyRobot
                            identifyMutation={identifyMutation}
                            robotForm={{
                                ...robotForm,
                                type: 'Trossen_WidowXAI_Follower',
                                connection_string: robotForm.connection_string_left,
                            }}
                        />
                    </View>
                </Flex>
            </Flex>

            <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
                <TextField
                    name='payload.connection_string_right'
                    isRequired
                    label='Right arm IP address'
                    width='100%'
                    value={robotForm.connection_string_right ?? ''}
                    onChange={(connection_string_right) => {
                        setRobotForm((oldForm) => ({
                            ...oldForm,
                            connection_string_right,
                            serial_number: '',
                        }));
                    }}
                    placeholder='192.168.1.3'
                />
                <View>
                    <IdentifyRobot
                        identifyMutation={identifyMutation}
                        robotForm={{
                            ...robotForm,
                            type: 'Trossen_WidowXAI_Follower',
                            connection_string: robotForm.connection_string_right,
                        }}
                    />
                </View>
            </Flex>
        </>
    );
};
