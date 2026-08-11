import { useEffect, useMemo, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Card,
    Checkbox,
    Content,
    ContextualHelp,
    Dialog,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Divider,
    Flex,
    Heading,
    Item,
    Key,
    NumberField,
    Picker,
    StatusLight,
    Text,
    View,
} from '@geti-ui/ui';

import { $api } from '../../api/client';
import {
    SchemaDeviceInfo,
    SchemaTrainJob as SchemaJob,
    SchemaModel,
    SchemaRemoteTrainerHealth,
} from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';
import { useRemoteTrainerHealth } from '../../features/remote-trainers/use-remote-trainer-health';
import { InlineAlert } from '../../features/robots/setup-wizard/shared/inline-alert';

import classes from './train-model-dialog.module.css';

export type SchemaTrainJob = Omit<SchemaJob, 'payload'> & {
    payload: SchemaJob['payload'];
};

export type TrainingEngine = SchemaJob['payload']['training_engine'];

type ExportBackend = 'torch' | 'openvino' | 'onnx' | 'executorch';

/**
 * Camera slots each multi-camera policy expects, in order. These are the
 * feature names baked into the pretrained policy configs (SmolVLA:
 * `camera1/2/3`; Pi0.5: `base_0_rgb` + left/right wrist). The dialog lets the
 * user map each slot to a dataset camera, or leave it empty.
 */
const CAMERA_SLOTS: Record<string, ReadonlyArray<{ label: string; name: string }>> = {
    smolvla: [
        { label: 'Camera 1', name: 'camera1' },
        { label: 'Camera 2', name: 'camera2' },
        { label: 'Camera 3', name: 'camera3' },
    ],
    pi05: [
        { label: 'Base', name: 'base_0_rgb' },
        { label: 'Left wrist', name: 'left_wrist_0_rgb' },
        { label: 'Right wrist', name: 'right_wrist_0_rgb' },
    ],
};

/** Formats a training job can export after training. */
const EXPORT_FORMATS: ReadonlyArray<{ id: ExportBackend; label: string }> = [
    { id: 'torch', label: 'PyTorch' },
    { id: 'openvino', label: 'OpenVINO' },
    { id: 'onnx', label: 'ONNX' },
    { id: 'executorch', label: 'ExecuTorch' },
];

const GB = 1024 ** 3;

/** Format bytes as a human-readable GB string. */
const formatBytes = (bytes: number): string => {
    const gb = bytes / GB;
    return gb >= 10 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`;
};

/**
 * Available training policies with hardware requirements.
 *
 * `minVRAM` is the estimated minimum VRAM (in bytes) required to train with batch_size=1.
 * `engines` lists the training engines that can run the policy.
 */
export const MODELS: ReadonlyArray<{
    id: string;
    name: string;
    description: string;
    minVRAM: number;
    engines: ReadonlyArray<TrainingEngine>;
}> = [
    {
        id: 'act',
        name: 'ACT',
        description: 'Action Chunking with Transformers, lightweight and fast to train',
        minVRAM: 2 * GB,
        engines: ['physicalai', 'lerobot'],
    },
    {
        id: 'diffusion',
        name: 'Diffusion Policy',
        description: 'Diffusion-based action generation, trained with the LeRobot engine',
        minVRAM: 8 * GB,
        engines: ['lerobot'],
    },
    {
        id: 'smolvla',
        name: 'SmolVLA',
        description: 'Small Vision-Language-Action model based on SmolVLM2-500M',
        minVRAM: 8 * GB,
        engines: ['physicalai', 'lerobot'],
    },
    {
        id: 'pi05',
        name: 'Pi0.5',
        description: 'Enhanced Pi0 with discrete state encoding and longer context',
        minVRAM: 16 * GB,
        engines: ['physicalai', 'lerobot'],
    },
];

interface TrainModelDialogProps {
    baseModel?: SchemaModel;
    close: (job: SchemaJob | undefined) => void;
    defaultMaxEpochs?: number;
}

type TrainingTargetOption = {
    id: string;
    label: string;
};

interface PolicySelectionProps {
    selectedPolicy: string;
    onSelectionChange: (policy: string) => void;
    isDisabled?: boolean;
    trainingDevice: SchemaDeviceInfo | null;
    engine: TrainingEngine;
}

const PolicySelection = ({
    selectedPolicy,
    onSelectionChange,
    isDisabled,
    trainingDevice,
    engine,
}: PolicySelectionProps) => {
    const availableVram = trainingDevice?.memory ?? 0;

    const engineModels = MODELS.filter((model) => model.engines.includes(engine));
    const selectedModel = engineModels.find((m) => m.id === selectedPolicy) ?? null;
    const hasInsufficientVram = selectedModel !== null && availableVram > 0 && selectedModel.minVRAM > availableVram;

    return (
        <Flex direction='column' gap='size-100'>
            <Text UNSAFE_style={{ fontSize: 12 }}>Policy</Text>
            <div className={classes.policyGrid}>
                {engineModels.map((model) => {
                    const isSelected = selectedPolicy === model.id;
                    if (isDisabled && !isSelected) {
                        return null;
                    }

                    return (
                        <Card
                            key={model.id}
                            aria-label={`Select ${model.name} policy`}
                            isSelected={isSelected}
                            isDisabled={isDisabled}
                            onPress={() => onSelectionChange(model.id)}
                            UNSAFE_className={classes.modelPolicyCard}
                        >
                            <Flex direction='column' gap='size-100'>
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
                                        <Text>&ge; {formatBytes(model.minVRAM)} VRAM</Text>
                                    </Flex>
                                </Flex>
                                <Divider size='S' />
                                <Text UNSAFE_style={{ fontSize: 12 }}>{model.description}</Text>
                            </Flex>
                        </Card>
                    );
                })}
            </div>

            {hasInsufficientVram && (
                <View marginTop='size-100'>
                    <InlineAlert variant='warning'>
                        {selectedModel.name} requires at least {formatBytes(selectedModel!.minVRAM)} VRAM but your
                        device has {formatBytes(availableVram)}. Training may fail or be very slow.
                    </InlineAlert>
                </View>
            )}
        </Flex>
    );
};

/** Pick the device with the most VRAM (if any) from a list of reported devices. */
const pickBestDevice = (devices: SchemaDeviceInfo[]): SchemaDeviceInfo | null =>
    devices
        .filter((d) => d.type !== 'cpu' && d.memory != null)
        .reduce((best: SchemaDeviceInfo | null, device) => {
            if (best === null || (device.memory ?? 0) > (best.memory ?? 0)) {
                return device;
            }

            return best;
        }, null);

const useBestTrainingDevice = (): SchemaDeviceInfo | null => {
    const { devices } = useTrainingDevices();

    return useMemo(() => pickBestDevice(devices), [devices]);
};

/**
 * Reads the training devices endpoint and normalizes the response.
 *
 * The endpoint always reports this Studio host's local training devices. Remote
 * trainer configuration is selected independently for each submitted job.
 */
const useTrainingDevices = () => {
    const { data } = $api.useQuery('get', '/api/system/devices/training', {}, { refetchOnMount: 'always' });

    return {
        devices: data?.devices ?? [],
    };
};

interface TrainingDeviceInfoProps {
    isRemoteTarget: boolean;
    remoteHealth: SchemaRemoteTrainerHealth | null;
    isCheckingRemote: boolean;
}

const TrainingDeviceInfo = ({ isRemoteTarget, remoteHealth, isCheckingRemote }: TrainingDeviceInfoProps) => {
    const bestDevice = useBestTrainingDevice();
    const bestRemoteDevice = useMemo(() => pickBestDevice(remoteHealth?.devices ?? []), [remoteHealth]);

    return (
        <Flex UNSAFE_style={{ textAlign: 'right' }} direction='column' gap='size-75'>
            {isRemoteTarget ? (
                remoteHealth?.status === 'unreachable' ? (
                    <StatusLight variant='negative'>Remote trainer unavailable</StatusLight>
                ) : bestRemoteDevice ? (
                    <StatusLight variant='positive'>
                        {bestRemoteDevice.name}, {formatBytes(bestRemoteDevice.memory!)} VRAM
                    </StatusLight>
                ) : isCheckingRemote && remoteHealth === null ? (
                    <StatusLight variant='neutral'>Checking remote trainer…</StatusLight>
                ) : (
                    <StatusLight variant='neutral'>Remote trainer selected</StatusLight>
                )
            ) : bestDevice ? (
                <StatusLight variant='positive'>
                    {bestDevice.name}, {formatBytes(bestDevice.memory!)} VRAM
                </StatusLight>
            ) : (
                <StatusLight variant='neutral'>CPU only (no GPU detected)</StatusLight>
            )}
        </Flex>
    );
};

const RECOMMENDED_PRECISION: Record<string, string> = {
    cuda: 'bf16-mixed',
};

const PRECISION_LABELS: Record<string, string> = {
    'bf16-mixed': 'BF16 Mixed',
    'bf16-true': 'BF16 True',
    '32-true': '32-bit',
};

interface TrainingParametersProps {
    maxEpochs: number;
    onMaxEpochsChange: (value: number) => void;
    batchSize: number;
    onBatchSizeChange: (value: number) => void;
    numWorkers: Key | null;
    onNumWorkersChange: (value: Key | null) => void;
    autoScaleBatchSize: boolean;
    onAutoScaleBatchSizeChange: (value: boolean) => void;
    precision: Key | null;
    onPrecisionChange: (value: Key | null) => void;
    compileModel: boolean;
    onCompileModelChange: (value: boolean) => void;
    isAutoScaleBatchDisabled: boolean;
    deviceType: string | undefined;
    engine: TrainingEngine;
}

const TrainingParameters = ({
    maxEpochs,
    onMaxEpochsChange,
    batchSize,
    onBatchSizeChange,
    numWorkers,
    onNumWorkersChange,
    autoScaleBatchSize,
    onAutoScaleBatchSizeChange,
    precision,
    onPrecisionChange,
    compileModel,
    onCompileModelChange,
    isAutoScaleBatchDisabled,
    deviceType,
    engine,
}: TrainingParametersProps) => {
    const isLerobot = engine === 'lerobot';

    return (
        <Flex direction='column' gap='size-150' width='100%'>
            {isLerobot && (
                <Text UNSAFE_style={{ fontSize: 12, opacity: 0.7 }}>
                    LeRobot uses the step budget and batch size set above, and manages precision and compilation
                    automatically. Auto scaling the batch size is supported.
                </Text>
            )}
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
                        <Checkbox
                            isSelected={autoScaleBatchSize}
                            onChange={onAutoScaleBatchSizeChange}
                            isDisabled={isAutoScaleBatchDisabled}
                        >
                            Auto scale batch size
                        </Checkbox>
                        <ContextualHelp variant='info'>
                            <Heading>Auto scale batch size</Heading>
                            <Content>
                                <Text>
                                    Automatically finds the largest batch size that fits in GPU memory before training
                                    starts. On XPU auto batch size is disabled.
                                </Text>
                            </Content>
                        </ContextualHelp>
                    </Flex>
                </Flex>
                <NumberField
                    label='Max Epochs'
                    value={maxEpochs}
                    onChange={onMaxEpochsChange}
                    minValue={1}
                    maxValue={1000}
                    step={1}
                    width='100%'
                    contextualHelp={
                        <ContextualHelp variant='info'>
                            <Heading>Max epochs</Heading>
                            <Content>
                                <Text>
                                    Total number of training epochs. Training will stop after this many full passes
                                    through the dataset. We recommend training for 5 to 10 epochs
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
                                    Number of parallel processes for loading training data. Auto selects a value based
                                    on available CPU cores. More workers can speed up training but use more memory.
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
            <Flex direction='row' gap='size-150' width='100%'>
                <Picker
                    width='100%'
                    label='Precision'
                    isDisabled={isLerobot}
                    description={
                        deviceType
                            ? `${
                                  PRECISION_LABELS[RECOMMENDED_PRECISION[deviceType] ?? '32-true']
                              } recommended for ${deviceType.toUpperCase()}`
                            : undefined
                    }
                    selectedKey={precision}
                    onSelectionChange={onPrecisionChange}
                    contextualHelp={
                        <ContextualHelp variant='info'>
                            <Heading>Training precision</Heading>
                            <Content>
                                <Text>
                                    Controls numerical precision during training. BF16 Mixed uses half-precision where
                                    safe for faster training and lower memory usage. BF16 True runs entirely in BF16 for
                                    maximum speed. 32-bit uses full precision for maximum numerical stability.
                                </Text>
                            </Content>
                        </ContextualHelp>
                    }
                >
                    <Item key='bf16-mixed'>BF16 Mixed</Item>
                    <Item key='bf16-true'>BF16 True</Item>
                    <Item key='32-true'>32-bit</Item>
                </Picker>
                <Flex direction='column' gap='size-150' width='100%' justifyContent='center'>
                    <Flex direction='row' gap='size-100' alignItems='center'>
                        <Checkbox isSelected={compileModel} onChange={onCompileModelChange} isDisabled={isLerobot}>
                            Compile model
                        </Checkbox>
                        <ContextualHelp variant='info'>
                            <Heading>Compile model</Heading>
                            <Content>
                                <Text>
                                    Enables torch.compile for all policies. Can significantly speed up training after an
                                    initial compilation warmup, but increases startup time.
                                </Text>
                            </Content>
                        </ContextualHelp>
                    </Flex>
                </Flex>
            </Flex>
        </Flex>
    );
};

export const TrainModelDialog = ({ baseModel, close, defaultMaxEpochs = 5 }: TrainModelDialogProps) => {
    const bestDevice = useBestTrainingDevice();
    const { data: remoteTrainers = [] } = $api.useQuery('get', '/api/remote-trainers');
    // Continuing an existing model needs its checkpoint, which only this machine
    // has: the trainer protocol can receive a dataset but not a base checkpoint.
    // So a resumed run offers local training only.
    const canTrainRemotely = baseModel === undefined;
    const trainingTargetOptions: TrainingTargetOption[] = [
        { id: 'local', label: 'This machine (local)' },
        ...(canTrainRemotely
            ? remoteTrainers.map((remoteTrainer) => ({
                  id: remoteTrainer.id,
                  label: remoteTrainer.name,
              }))
            : []),
    ];

    const defaultDatasetId = baseModel?.dataset_id ?? null;
    const extraPayload = baseModel ? { base_model_id: baseModel.id! } : undefined;

    const [selectedPolicy, setSelectedPolicy] = useState<string>(baseModel?.policy ?? 'act');
    const [trainingEngine, setTrainingEngine] = useState<TrainingEngine>(
        baseModel?.properties?.training_engine === 'lerobot' ? 'lerobot' : 'physicalai'
    );
    const { datasets, id: projectId } = useProject();

    const [selectedDataset, setSelectedDataset] = useState<Key | null>(defaultDatasetId);
    const [maxEpochs, setMaxEpochs] = useState<number>(defaultMaxEpochs);
    const [batchSize, setBatchSize] = useState<number>(8);
    const [numWorkers, setNumWorkers] = useState<Key | null>('auto');
    const [autoScaleBatchSize, setAutoScaleBatchSize] = useState<boolean>(bestDevice?.type === 'cuda');
    const [precision, setPrecision] = useState<Key | null>(bestDevice?.type === 'cuda' ? 'bf16-mixed' : '32-true');
    const [compileModel, setCompileModel] = useState<boolean>(false);
    const [renameMap, setRenameMap] = useState<Record<string, string | null>>({});
    const [exportBackends, setExportBackends] = useState<ExportBackend[]>(['torch', 'openvino']);
    const [remoteTrainerId, setRemoteTrainerId] = useState<Key | null>('local');
    const isRemoteTarget = remoteTrainerId !== null && remoteTrainerId !== 'local';
    const {
        health: remoteTrainerHealth,
        isChecking: isCheckingRemoteTrainer,
        checkHealth: checkRemoteTrainerHealth,
    } = useRemoteTrainerHealth(isRemoteTarget ? (remoteTrainerId?.toString() ?? null) : null);
    const remoteUnavailable = isRemoteTarget && remoteTrainerHealth?.status === 'unreachable';
    const bestRemoteDevice = useMemo(() => pickBestDevice(remoteTrainerHealth?.devices ?? []), [remoteTrainerHealth]);
    // The device actually driving this job: the local GPU when training locally,
    // or the remote trainer's reported GPU once its health check resolves. Auto
    // scale/precision defaults and the disabled state below should track whichever
    // one is currently in play, the same way they did when there was only ever a
    // single active device to consider.
    const activeDevice = isRemoteTarget ? bestRemoteDevice : bestDevice;

    useEffect(() => {
        if (activeDevice?.type === 'cuda') {
            setPrecision('bf16-mixed');
            setAutoScaleBatchSize(true);
        } else {
            setPrecision('32-true');
            setAutoScaleBatchSize(false);
        }
    }, [activeDevice]);

    const datasetId = selectedDataset?.toString();
    const { data: datasetCameras = [] } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/cameras',
        {
            params: { path: { dataset_id: datasetId! } },
        },
        {
            enabled: datasetId !== undefined,
        }
    );

    // A new dataset or policy invalidates earlier camera selections.
    useEffect(() => {
        setRenameMap({});
    }, [selectedDataset, selectedPolicy]);

    const cameraSlots = CAMERA_SLOTS[selectedPolicy] ?? [];
    const cameraOptions = datasetCameras.map((cameraKey) => cameraKey.replace(/^observation\.images\./, ''));

    const trainMutation = $api.useMutation('post', '/api/jobs:train', {
        meta: {
            invalidates: [['get', '/api/jobs']],
        },
    });

    const handleTrainingEngineChange = (engine: Key | null) => {
        if (engine === null) {
            return;
        }
        const next = engine as TrainingEngine;
        setTrainingEngine(next);
        if (!MODELS.some((model) => model.engines.includes(next) && model.id === selectedPolicy)) {
            const fallback = MODELS.find((model) => model.engines.includes(next));
            if (fallback) {
                setSelectedPolicy(fallback.id);
            }
        }
    };

    const save = async () => {
        const dataset_id = selectedDataset?.toString();

        if (!dataset_id || !selectedPolicy || remoteTrainerId === null) {
            return;
        }

        if (isRemoteTarget) {
            // Final guard: the remote trainer may have gone offline since the last
            // poll, so re-check availability right before submitting the job.
            const latestHealth = await checkRemoteTrainerHealth();
            if (latestHealth === null || latestHealth.status === 'unreachable') {
                return;
            }
        }

        const name = baseModel?.name ?? MODELS.find((policy) => policy.id === selectedPolicy)?.name ?? '';

        const payload: SchemaJob['payload'] = {
            dataset_id,
            project_id: projectId,
            model_name: name,
            policy: selectedPolicy,
            max_epochs: maxEpochs,
            training_engine: trainingEngine,
            batch_size: batchSize,
            num_workers: numWorkers === 'auto' ? 'auto' : Number(numWorkers),
            auto_scale_batch_size: autoScaleBatchSize,
            precision: (precision?.toString() ?? 'bf16-mixed') as SchemaJob['payload']['precision'],
            compile_model: compileModel,
            val_split: 0.1,
            training_target: isRemoteTarget ? 'remote' : 'local',
            ...(Object.keys(renameMap).length > 0 ? { rename_map: renameMap } : {}),
            export_backends: exportBackends,
            ...(isRemoteTarget ? { remote_trainer_id: remoteTrainerId?.toString() } : {}),
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

                    <TrainingDeviceInfo
                        isRemoteTarget={isRemoteTarget}
                        remoteHealth={remoteTrainerHealth ?? null}
                        isCheckingRemote={isCheckingRemoteTrainer}
                    />
                </Flex>
            </Heading>
            <Divider />
            <Content width={'700px'}>
                <Flex direction='column' gap='size-200' width='100%'>
                    {remoteUnavailable && (
                        <InlineAlert variant='warning'>
                            Can&apos;t reach the remote trainer, so training can&apos;t start. Make sure it&apos;s
                            running, then try again.
                        </InlineAlert>
                    )}

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

                    <Picker
                        label='Run on'
                        selectedKey={remoteTrainerId}
                        onSelectionChange={setRemoteTrainerId}
                        width='100%'
                        items={trainingTargetOptions}
                    >
                        {(trainingTarget) => <Item key={trainingTarget.id}>{trainingTarget.label}</Item>}
                    </Picker>

                    <Picker
                        label='Training engine'
                        selectedKey={trainingEngine}
                        onSelectionChange={handleTrainingEngineChange}
                        isDisabled={baseModel !== undefined}
                        width='100%'
                        contextualHelp={
                            <ContextualHelp variant='info'>
                                <Heading>Training engine</Heading>
                                <Content>
                                    <Text>
                                        Which training stack to run. PhysicalAI uses the Lightning-based
                                        physicalai-train stack; LeRobot uses LeRobot&apos;s own training loop.
                                    </Text>
                                </Content>
                            </ContextualHelp>
                        }
                    >
                        <Item key='physicalai'>PhysicalAI (Lightning)</Item>
                        <Item key='lerobot'>LeRobot</Item>
                    </Picker>

                    <PolicySelection
                        selectedPolicy={selectedPolicy}
                        onSelectionChange={setSelectedPolicy}
                        isDisabled={baseModel !== undefined}
                        trainingDevice={activeDevice}
                        engine={trainingEngine}
                    />

                    {cameraSlots.length > 0 && (
                        <Flex direction='column' gap='size-150'>
                            <Text UNSAFE_style={{ fontSize: 12, fontWeight: 600 }}>Camera mapping</Text>
                            <Text UNSAFE_style={{ fontSize: 12, opacity: 0.7 }}>
                                Map each {MODELS.find((m) => m.id === selectedPolicy)?.name ?? 'model'} camera to a
                                camera recorded in your dataset, or leave it empty.
                            </Text>
                            {cameraSlots.map((slot) => {
                                const options = [
                                    { id: 'empty', label: 'Empty' },
                                    ...cameraOptions.map((camera) => ({ id: camera, label: camera })),
                                ];
                                return (
                                    <Picker
                                        key={slot.name}
                                        label={slot.label}
                                        selectedKey={renameMap[slot.name] ?? 'empty'}
                                        onSelectionChange={(key) =>
                                            setRenameMap((prev) => ({
                                                ...prev,
                                                [slot.name]: key === 'empty' ? null : String(key),
                                            }))
                                        }
                                        isDisabled={baseModel !== undefined}
                                        width='100%'
                                        items={options}
                                    >
                                        {(option) => <Item key={option.id}>{option.label}</Item>}
                                    </Picker>
                                );
                            })}
                        </Flex>
                    )}

                    <Disclosure
                        isQuiet
                        UNSAFE_style={{ padding: 0 }}
                        UNSAFE_className={classes.advancedSettingsDisclosure}
                        defaultExpanded={bestDevice?.type !== 'cuda'}
                    >
                        <DisclosureTitle UNSAFE_style={{ fontSize: 13, padding: '4px 0' }}>
                            Advanced settings
                        </DisclosureTitle>
                        <DisclosurePanel UNSAFE_style={{ padding: 0 }}>
                            <Flex direction='column' gap='size-150' marginBottom='size-150'>
                                <Text UNSAFE_style={{ fontSize: 12, fontWeight: 600 }}>Export formats</Text>
                                <Flex direction='row' gap='size-200' wrap>
                                    {EXPORT_FORMATS.map((format) => (
                                        <Checkbox
                                            key={format.id}
                                            isSelected={exportBackends.includes(format.id)}
                                            onChange={(checked) =>
                                                setExportBackends((prev) =>
                                                    checked
                                                        ? [...prev, format.id]
                                                        : prev.filter((backend) => backend !== format.id)
                                                )
                                            }
                                            isDisabled={baseModel !== undefined}
                                        >
                                            {format.label}
                                        </Checkbox>
                                    ))}
                                </Flex>
                            </Flex>
                            <TrainingParameters
                                maxEpochs={maxEpochs}
                                onMaxEpochsChange={setMaxEpochs}
                                batchSize={batchSize}
                                onBatchSizeChange={setBatchSize}
                                numWorkers={numWorkers}
                                onNumWorkersChange={setNumWorkers}
                                autoScaleBatchSize={autoScaleBatchSize}
                                onAutoScaleBatchSizeChange={setAutoScaleBatchSize}
                                precision={precision}
                                onPrecisionChange={setPrecision}
                                compileModel={compileModel}
                                onCompileModelChange={setCompileModel}
                                isAutoScaleBatchDisabled={activeDevice?.type !== 'cuda'}
                                deviceType={activeDevice?.type}
                                engine={trainingEngine}
                            />
                        </DisclosurePanel>
                    </Disclosure>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={save}
                    isDisabled={!selectedDataset || !selectedPolicy || remoteTrainerId === null || remoteUnavailable}
                >
                    Train
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
