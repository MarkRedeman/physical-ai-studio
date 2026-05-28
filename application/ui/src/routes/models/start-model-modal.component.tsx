import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading } from '@geti-ui/ui';
import { useNavigate } from 'react-router';
import { createSearchParams } from 'react-router-dom';

import { $api } from '../../api/client';
import { SchemaModel } from '../../api/openapi-spec';
import {
    BackendSelection,
    defaultBackend,
    getDefaultInferenceDevice,
    getSupportedInferenceDevices,
} from '../../features/models/backend-selection';
import { paths } from '../../router';

const getDefaultbackend = (model: SchemaModel) => {
    if (model.available_backends.includes(defaultBackend)) {
        return defaultBackend;
    }

    return model.available_backends.at(0) ?? defaultBackend;
};

interface StartInferenceDialogProps {
    close: () => void;
    model: SchemaModel;
}

export const StartInferenceDialog = ({ close, model }: StartInferenceDialogProps) => {
    const [backend, setBackend] = useState<string>(getDefaultbackend(model));
    const [device, setDevice] = useState<string | undefined>();

    const { data: inferenceDevices = [], isLoading } = $api.useQuery('get', '/api/system/devices/inference');

    const selectedDevice = getSupportedInferenceDevices(inferenceDevices, backend).find(
        (inferenceDevice) => inferenceDevice.device === device
    );
    const inferenceDevice = selectedDevice ?? getDefaultInferenceDevice(inferenceDevices, backend);

    const navigate = useNavigate();
    const onStart = () => {
        if (inferenceDevice === undefined) {
            return;
        }

        close();
        navigate({
            pathname: paths.project.models.inference({
                project_id: model.project_id,
                model_id: model.id!,
                backend,
            }),
            search: createSearchParams({ device: inferenceDevice.device }).toString(),
        });
    };

    return (
        <Dialog>
            <Heading>Select your inference backend</Heading>
            <Divider />
            <Content>
                <BackendSelection
                    model={model}
                    backend={backend}
                    device={inferenceDevice?.device}
                    inferenceDevices={inferenceDevices}
                    setBackend={setBackend}
                    setDevice={setDevice}
                />
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={onStart} isDisabled={isLoading || inferenceDevice === undefined}>
                    Start
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
