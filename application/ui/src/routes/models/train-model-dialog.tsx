import { useMemo, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Card,
    Checkbox,
    Content,
    ContextualHelp,
    Dialog,
    Divider,
    Flex,
    Form,
    Heading,
    Item,
    Key,
    NumberField,
    Picker,
    StatusLight,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';

import { $api } from '../../api/client';
import { SchemaDeviceInfo, SchemaModel, SchemaTrainJob, SchemaTrainJobPayload } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';
import { InlineAlert } from '../../features/robots/setup-wizard/shared/inline-alert';

import classes from './train-model-dialog.module.scss';

const GB = 1024 ** 3;

/** Format bytes as a human-readable GB string. */
const formatBytes = (bytes: number): string => {
    const gb = bytes / GB;
    return gb >= 10 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`;
};

/**
 * Available training policies with hardware requirements.
 *
 * `min_vram` is the estimated minimum VRAM (in bytes) required to train with batch_size=1.
 */
export const MODELS: ReadonlyArray<{
    id: string;
    name: string;
    description: string;
    license: string;
    min_episodes: number | null;
    min_steps: number | null;
    min_vram: number;
}> = [
    {
        id: 'act',
        name: 'ACT',
        description: 'Action Chunking with Transformers, lightweight and fast to train',
        license: 'MIT / Apache-2.0',
        min_episodes: null,
        min_steps: null,
        min_vram: 2 * GB,
    },
    {
        id: 'smolvla',
        name: 'SmolVLA',
        description: 'Small Vision-Language-Action model based on SmolVLM2-500M',
        license: 'Apache-2.0',
        min_episodes: null,
        min_steps: null,
        min_vram: 8 * GB,
    },
    {
        id: 'pi05',
        name: 'Pi0.5',
        description: 'Enhanced Pi0 with discrete state encoding and longer context',
        license: 'Apache-2.0 (code) / Gemma (weights)',
        min_episodes: null,
        min_steps: null,
        min_vram: 16 * GB,
    },
];

export type { SchemaTrainJob };

interface TrainModelDialogProps {
    baseModel?: SchemaModel;
    close: (job: SchemaTrainJob | undefined) => void;
    defaultMaxSteps?: number;
}

interface PolicySelectionProps {
    selectedPolicy: string;
    onSelectionChange: (policy: string) => void;
    isDisabled?: boolean;
    availableVram: number;
}

const PolicySelection = ({ selectedPolicy, onSelectionChange, isDisabled, availableVram }: PolicySelectionProps) => {
    const selectedModel = MODELS.find((m) => m.id === selectedPolicy) ?? null;
    const hasInsufficientVram = selectedModel !== null && availableVram > 0 && selectedModel.min_vram > availableVram;

    return (
        <Flex direction='column' gap='size-100'>
            <Text UNSAFE_style={{ fontSize: 12, fontWeight: 600 }}>Policy</Text>
            <Flex direction='row' gap='size-200'>
                {MODELS.map((model) => (
                    <Card
                        key={model.id}
                        aria-label={`Select ${model.name} policy`}
                        isSelected={selectedPolicy === model.id}
                        isDisabled={isDisabled}
                        onPress={() => onSelectionChange(model.id)}
                        UNSAFE_className={classes.modelPolicyCard}
                    >
                        <Flex direction='column' gap='size-50'>
                            <Flex justifyContent={'space-between'}>
                                <Text
                                    UNSAFE_style={{
                                        fontWeight: 700,
                                        color: selectedPolicy === model.id ? 'var(--energy-blue)' : undefined,
                                    }}
                                >
                                    {model.name}
                                </Text>
                                <Flex
                                    UNSAFE_style={{ fontSize: 11, opacity: 0.7, textAlign: 'right' }}
                                    direction='column'
                                    gap='size-50'
                                >
                                    <Text>{model.license}</Text>
                                    <Text>&ge; {formatBytes(model.min_vram)} VRAM</Text>
                                </Flex>
                            </Flex>
                            <Divider size='S' />
                            <Text UNSAFE_style={{ fontSize: 12 }}>{model.description}</Text>
                        </Flex>
                    </Card>
                ))}
            </Flex>

            {hasInsufficientVram && (
                <View marginTop='size-100'>
                    <InlineAlert variant='warning'>
                        {selectedModel!.name} requires at least {formatBytes(selectedModel!.min_vram)} VRAM but your
                        device has {formatBytes(availableVram)}. Training may fail or be very slow.
                    </InlineAlert>
                </View>
            )}
        </Flex>
    );
};

interface TrainingParametersProps {
    maxSteps: number;
    onMaxStepsChange: (value: number) => void;
    batchSize: number;
    onBatchSizeChange: (value: number) => void;
    numWorkers: Key | null;
    onNumWorkersChange: (value: Key | null) => void;
    autoScaleBatchSize: boolean;
    onAutoScaleBatchSizeChange: (value: boolean) => void;
}

const TrainingParameters = ({
    maxSteps,
    onMaxStepsChange,
    batchSize,
    onBatchSizeChange,
    numWorkers,
    onNumWorkersChange,
    autoScaleBatchSize,
    onAutoScaleBatchSizeChange,
}: TrainingParametersProps) => (
    <Flex direction='row' gap='size-150' width='100%'>
        <Flex direction='column' gap='size-150' width='100%'>
            <NumberField
                label='Batch Size'
                value={batchSize}
                onChange={onBatchSizeChange}
                minValue={1}
                maxValue={256}
                step={1}
                width='100%'
                isDisabled={autoScaleBatchSize}
                flex
            />
            <Flex direction='row' gap='size-100' alignItems='center'>
                <Checkbox isSelected={autoScaleBatchSize} onChange={onAutoScaleBatchSizeChange}>
                    Auto scale batch size
                </Checkbox>
                <ContextualHelp variant='info'>
                    <Heading>Auto scale batch size</Heading>
                    <Content>
                        <Text>
                            Automatically finds the largest batch size that fits in GPU memory before training starts.
                        </Text>
                    </Content>
                </ContextualHelp>
            </Flex>
        </Flex>
        <NumberField
            label='Max Steps'
            value={maxSteps}
            onChange={onMaxStepsChange}
            minValue={100}
            maxValue={100000}
            step={100}
            width='100%'
            contextualHelp={
                <ContextualHelp variant='info'>
                    <Heading>Max steps</Heading>
                    <Content>
                        <Text>
                            Total number of gradient update steps. Training will stop after this many steps regardless
                            of epochs.
                        </Text>
                    </Content>
                </ContextualHelp>
            }
        />
        <Picker
            width='100%'
            label='Data Workers'
            selectedKey={numWorkers}
            onSelectionChange={onNumWorkersChange}
            contextualHelp={
                <ContextualHelp variant='info'>
                    <Heading>Data workers</Heading>
                    <Content>
                        <Text>
                            Number of parallel processes for loading training data. Auto selects a value based on
                            available CPU cores. More workers can speed up training but use more memory.
                        </Text>
                    </Content>
                </ContextualHelp>
            }
        >
            <Item key='auto'>Auto</Item>
            <Item key='0'>0 (main process)</Item>
            <Item key='1'>1</Item>
            <Item key='2'>2</Item>
            <Item key='4'>4</Item>
            <Item key='8'>8</Item>
            <Item key='16'>16</Item>
        </Picker>
    </Flex>
);

export const TrainModelDialog = ({ baseModel, close, defaultMaxSteps = 10000 }: TrainModelDialogProps) => {
    const defaultName = baseModel?.name ?? '';
    const defaultDatasetId = baseModel?.dataset_id ?? null;
    const extraPayload = baseModel ? { base_model_id: baseModel.id! } : undefined;

    const [selectedPolicy, setSelectedPolicy] = useState<string>(baseModel?.policy ?? 'act');
    const { datasets, id: projectId } = useProject();

    const [name, setName] = useState<string>(defaultName);
    const [selectedDataset, setSelectedDataset] = useState<Key | null>(defaultDatasetId);
    const [maxSteps, setMaxSteps] = useState<number>(defaultMaxSteps);
    const [batchSize, setBatchSize] = useState<number>(8);
    const [numWorkers, setNumWorkers] = useState<Key | null>('auto');
    const [autoScaleBatchSize, setAutoScaleBatchSize] = useState<boolean>(true);

    // Fetch training devices from system endpoint
    const { data: trainingDevices = null } = $api.useQuery('get', '/api/system/devices/training');

    // Pick the GPU with the most VRAM (if any)
    const bestDevice: SchemaDeviceInfo | null = useMemo(() => {
        if (!trainingDevices) return null;
        const gpuDevices = trainingDevices.filter((d) => d.type !== 'cpu' && d.memory != null);
        if (gpuDevices.length === 0) return null;
        return gpuDevices.reduce((best, d) => ((d.memory ?? 0) > (best.memory ?? 0) ? d : best));
    }, [trainingDevices]);

    const availableVram = bestDevice?.memory ?? 0;

    const trainMutation = $api.useMutation('post', '/api/jobs:train');

    const save = () => {
        const dataset_id = selectedDataset?.toString();

        if (!dataset_id || !selectedPolicy) {
            return;
        }

        const payload: SchemaTrainJobPayload = {
            dataset_id,
            project_id: projectId,
            model_name: name,
            policy: selectedPolicy,
            max_steps: maxSteps,
            batch_size: batchSize,
            num_workers: numWorkers === 'auto' ? 'auto' : Number(numWorkers),
            auto_scale_batch_size: autoScaleBatchSize,
            device: bestDevice ? { type: bestDevice.type, index: bestDevice.index ?? undefined } : undefined,
            ...extraPayload,
        };
        trainMutation.mutateAsync({ body: payload }).then((response) => {
            close(response as SchemaTrainJob | undefined);
        });
    };

    return (
        <Dialog size='L' UNSAFE_style={{ width: 'fit-content' }}>
            <Heading>
                <Flex justifyContent={'space-between'}>
                    <Text> Train model</Text>

                    {/* Training device info */}
                    {trainingDevices !== null && (
                        <Flex UNSAFE_style={{ textAlign: 'right' }} direction='column' gap='size-75'>
                            {bestDevice ? (
                                <StatusLight variant='positive'>
                                    {bestDevice.name}, {formatBytes(bestDevice.memory!)} VRAM
                                </StatusLight>
                            ) : (
                                <StatusLight variant='neutral'>CPU only (no GPU detected)</StatusLight>
                            )}
                        </Flex>
                    )}
                </Flex>
            </Heading>
            <Divider />
            <Content width={'800px'}>
                <Form
                    onSubmit={(e) => {
                        e.preventDefault();
                        save();
                    }}
                    validationBehavior='native'
                >
                    <Flex direction='column' gap='size-200' width='100%'>
                        <TextField label='Name' value={name} onChange={setName} width='100%' />

                        <Picker
                            label='Dataset'
                            selectedKey={selectedDataset}
                            onSelectionChange={setSelectedDataset}
                            width='100%'
                        >
                            {datasets.map((dataset) => (
                                <Item key={dataset.id}>{dataset.name}</Item>
                            ))}
                        </Picker>

                        <PolicySelection
                            selectedPolicy={selectedPolicy}
                            onSelectionChange={setSelectedPolicy}
                            isDisabled={baseModel !== undefined}
                            availableVram={availableVram}
                        />

                        <Divider size='S' />

                        <TrainingParameters
                            maxSteps={maxSteps}
                            onMaxStepsChange={setMaxSteps}
                            batchSize={batchSize}
                            onBatchSizeChange={setBatchSize}
                            numWorkers={numWorkers}
                            onNumWorkersChange={setNumWorkers}
                            autoScaleBatchSize={autoScaleBatchSize}
                            onAutoScaleBatchSizeChange={setAutoScaleBatchSize}
                        />
                    </Flex>
                </Form>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={save}>
                    Train
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
