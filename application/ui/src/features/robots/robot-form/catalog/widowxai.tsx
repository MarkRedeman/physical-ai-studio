import { Flex, TextField } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import { buildRobotBody } from '../form-data';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot, useIdentifyMutation } from './actions';

export interface WidowxFormData {
    name: string;
    connection_string: string;
    serial_number: string;
}

export const getInitialWidowxFormData = (existing?: Partial<WidowxFormData>): WidowxFormData => ({
    name: existing?.name ?? '',
    connection_string: existing?.connection_string ?? '',
    serial_number: existing?.serial_number ?? '',
});

export const WidowxAIFormFields = () => {
    const { formData, updateField, activeType } = useRobotFormFields('widowx');

    const identifyMutation = useIdentifyMutation();
    const identifyRobot = buildRobotBody('widowx', formData, activeType, uuidv4());

    return (
        <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
            <TextField
                isRequired
                label='Robot IP address'
                width='100%'
                value={formData.connection_string}
                onChange={(connection_string) => {
                    updateField('connection_string', connection_string);
                    updateField('serial_number', '');
                }}
                placeholder='192.168.1.2'
            />
            <Flex gap='size-100'>
                <IdentifyRobot identifyMutation={identifyMutation} robot={identifyRobot} />
            </Flex>
        </Flex>
    );
};
