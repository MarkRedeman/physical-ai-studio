import { Flex, TextField } from '@geti/ui';

import { SchemaWebsocketCameraInput } from '../../../../api/openapi-spec';
import { NameField } from '../components/name-field';
import { useCameraFormFields } from '../components/use-camera-form-fields';
import { DriverFormSchema } from '../provider';

export const initialWebsocketState: DriverFormSchema<'websocket'> = {
    driver: 'websocket',
    hardware_name: null,
    payload: {
        fps: 30,
        width: 640,
        height: 480,
        websocket_url: '',
    },
};

export const validateWebsocket = (formData: DriverFormSchema<'websocket'>): formData is SchemaWebsocketCameraInput => {
    return (
        !!formData.name &&
        !!formData.fingerprint &&
        !!formData.payload?.width &&
        !!formData.payload?.height &&
        !!formData.payload?.fps &&
        !!formData.payload?.websocket_url
    );
};

export const WebsocketFormFields = () => {
    const { formData, updateField, updatePayload } = useCameraFormFields('websocket');

    return (
        <Flex gap='size-100' alignItems='end' direction='column'>
            <NameField value={formData.name ?? ''} onChange={(name) => updateField('name', name)} />
            <TextField
                isRequired
                label='Stream URL'
                width='100%'
                value={formData.fingerprint ?? ''}
                onChange={(url) => {
                    updateField('fingerprint', url);
                    updatePayload({ websocket_url: url });
                }}
            />
        </Flex>
    );
};
