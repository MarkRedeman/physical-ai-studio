import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading, Item, Picker, Slider } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaModel } from '../../../api/openapi-spec';
import { getDefaultInferenceDevice, getSupportedInferenceDevices } from '../backend-selection/backend-selection';
import { runtimeExportUrl } from '../runtime-export';
import { useVariantSelection, VariantSelection } from './variant-selection';

interface RuntimeExportDialogProps {
    close: () => void;
    model: SchemaModel;
    backend: string;
    initialVariant?: string;
}

export const RuntimeExportDialog = ({ close, model, backend, initialVariant }: RuntimeExportDialogProps) => {
    const selection = useVariantSelection(model.id, backend, initialVariant);
    const [horizon, setHorizon] = useState(1);
    const [guidance, setGuidance] = useState(5);
    const chunkSize =
        selection.selected?.io_spec?.output_features.find((feature) => feature.name === 'action')?.shape?.[0] ?? 2;
    const maxHorizon = Math.max(1, Math.floor(chunkSize / 2));
    const effectiveHorizon = Math.min(horizon, maxHorizon);
    const { data: environments = [] } = $api.useQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id: model.project_id } },
    });
    const { data: dataset } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        { params: { path: { dataset_id: model.dataset_id! } } },
        { enabled: model.dataset_id != null }
    );
    const { data: tasks = [] } = $api.useQuery(
        'get',
        '/api/models/{model_id}/tasks',
        { params: { path: { model_id: model.id! } } },
        { enabled: model.dataset_id != null }
    );
    const { data: inferenceDevices = [] } = $api.useQuery('get', '/api/system/devices/inference');

    const defaultEnvironmentId = dataset?.environment_id ?? environments[0]?.id;
    const [environmentId, setEnvironmentId] = useState<string | undefined>(defaultEnvironmentId);
    const selectedEnvironmentId = environmentId ?? defaultEnvironmentId;

    const devices = getSupportedInferenceDevices(inferenceDevices, backend);
    const defaultDevice = getDefaultInferenceDevice(inferenceDevices, backend);
    const [device, setDevice] = useState<string | undefined>(defaultDevice?.device);
    const selectedDevice = device ?? defaultDevice?.device;

    const [task, setTask] = useState<string | undefined>(tasks[0]);
    const selectedTask = task ?? tasks[0];

    const downloadUrl =
        model.id !== undefined && selectedEnvironmentId !== undefined && selectedDevice !== undefined
            ? runtimeExportUrl({
                  modelId: model.id,
                  environmentId: selectedEnvironmentId,
                  backend,
                  device: selectedDevice,
                  task: selectedTask,
                  variant: selection.variant,
                  executionHorizon: selection.selected?.rtc_enabled ? effectiveHorizon : undefined,
                  maxGuidanceWeight: selection.selected?.rtc_enabled ? guidance : undefined,
              })
            : undefined;

    return (
        <Dialog>
            <Heading>Download runtime export</Heading>
            <Divider />
            <Content>
                {selection.variants.length > 0 && (
                    <VariantSelection
                        variants={selection.variants}
                        variant={selection.variant}
                        onChange={selection.setVariant}
                    />
                )}
                {selection.selected?.rtc_enabled && (
                    <>
                        <Slider
                            label='RTC execution horizon'
                            minValue={1}
                            maxValue={maxHorizon}
                            step={1}
                            value={effectiveHorizon}
                            onChange={setHorizon}
                        />
                        <Slider
                            label='RTC guidance weight'
                            minValue={0}
                            maxValue={10}
                            step={0.5}
                            value={guidance}
                            onChange={setGuidance}
                        />
                    </>
                )}
                <Picker
                    isRequired
                    label='Environment'
                    items={environments}
                    selectedKey={selectedEnvironmentId}
                    onSelectionChange={(key) => setEnvironmentId(key === null ? undefined : String(key))}
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>
                <Picker
                    isRequired
                    label='Device'
                    items={devices}
                    selectedKey={selectedDevice}
                    onSelectionChange={(key) => setDevice(key === null ? undefined : String(key))}
                >
                    {(item) => <Item key={item.device}>{item.name}</Item>}
                </Picker>
                {tasks.length > 0 && (
                    <Picker
                        label='Task'
                        items={tasks.map((value) => ({ id: value, name: value }))}
                        selectedKey={selectedTask}
                        onSelectionChange={(key) => setTask(key === null ? undefined : String(key))}
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </Picker>
                )}
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    href={downloadUrl}
                    isDisabled={downloadUrl === undefined || selection.isPending || selection.isError}
                    onPress={close}
                    target='_blank'
                    rel='noopener noreferrer'
                >
                    Download
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
