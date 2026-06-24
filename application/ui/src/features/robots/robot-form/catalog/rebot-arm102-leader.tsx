import { Flex, TextField } from '@geti-ui/ui';

import { useRobotForm, useSetRobotForm } from '../provider';
import { IdentifyRobot, useIdentifyMutation } from './actions';

export const ReBotArm102LeaderFormFields = () => {
    const robotForm = useRobotForm();
    const setRobotForm = useSetRobotForm();

    const identifyMutation = useIdentifyMutation();

    return (
        <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
            <TextField
                isRequired
                label='Robot connection string'
                width='100%'
                value={robotForm.connection_string ?? ''}
                onChange={(connection_string) => {
                    setRobotForm((oldForm) => ({
                        ...oldForm,
                        connection_string,
                        serial_number: '',
                    }));
                }}
                placeholder='/dev/ttyACM0'
            />
            <Flex gap='size-100'>
                <IdentifyRobot identifyMutation={identifyMutation} robotForm={robotForm} />
            </Flex>
        </Flex>
    );
};
