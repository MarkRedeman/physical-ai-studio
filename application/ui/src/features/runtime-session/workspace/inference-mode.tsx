import { createContext, ReactNode, useContext, useEffect, useState } from 'react';

import {
    Button,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Radio,
    RadioGroup,
    Slider,
    Text,
    TextArea,
} from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaExecutionConfig, SchemaModelControlConfig, SchemaSessionModel } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS } from '../../models/inference-backends';
import { failureOf, useRuntimeSession } from '../runtime-session-provider';
import { usePartControls } from './part-controls';
import { PartRow } from './parts-panel';
import { useSessionMode } from './session-mode';

import settingsClasses from './inference-mode.module.css';
import classes from './record-mode.module.css';

const EXECUTION_MODES: { key: SchemaExecutionConfig['mode']; label: string }[] = [
    { key: 'sync', label: 'Synchronous' },
    { key: 'async', label: 'Asynchronous' },
];

/** The default task of the dataset the loaded model was trained on, or none. */
const useDefaultTask = (model_id: string | undefined) => {
    const { data: model } = $api.useQuery(
        'get',
        '/api/models/{model_id}',
        {
            params: { path: { model_id: model_id ?? '' } },
        },
        { enabled: model_id !== undefined }
    );
    const dataset_id = model?.model.dataset_id ?? undefined;
    const { data: dataset } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        {
            params: { path: { dataset_id: dataset_id ?? '' } },
        },
        { enabled: dataset_id !== undefined }
    );
    return dataset?.default_task ?? '';
};

const useInferenceState = () => {
    const { state, statuses, isLoading } = useRuntimeSession();
    const { pendingPart } = usePartControls();
    const { mode: selectedMode, run, isPending } = useSessionMode();
    const defaultTask = useDefaultTask(state?.model?.model_id);
    const [task, setTask] = useState(defaultTask);
    const [executionMode, setExecutionMode] = useState<SchemaExecutionConfig['mode']>('sync');
    const [threshold, setThreshold] = useState(0.5);
    // Use the model's training fps once exports record it.
    const [hz, setHz] = useState(30);
    useEffect(() => setTask(defaultTask), [defaultTask]);

    const isRunning = state?.control?.type === 'model';
    const isRecording = state?.dataset?.is_recording ?? false;
    const isLocked = isRunning || isRecording || isPending || isLoading || pendingPart !== null;
    const canRun =
        selectedMode === 'inference' &&
        state?.environment != null &&
        failureOf(statuses, 'environment') === null &&
        state.model != null &&
        failureOf(statuses, 'model') === null &&
        !isLocked;

    const play = () => {
        if (!canRun) return;
        const control: SchemaModelControlConfig = {
            type: 'model',
            task,
            hz,
            execution: { mode: executionMode, request_threshold: threshold },
            // Ease onto model actions instead of jumping from the follower's current position.
            grace_s: 3,
        };
        void run({ command: 'set_control', control, reject_open_episode: true }, 'Could not run the model.');
    };
    const stop = () => {
        if (isRunning && !isPending && !isRecording)
            void run({ command: 'set_control', control: null, reject_open_episode: true }, 'Could not stop the model.');
    };

    return {
        task,
        setTask,
        executionMode,
        setExecutionMode,
        threshold,
        setThreshold,
        hz,
        setHz,
        isRunning,
        isRecording,
        isLocked,
        isPending,
        canRun,
        play,
        stop,
    };
};

const InferenceContext = createContext<ReturnType<typeof useInferenceState> | null>(null);

export const InferenceProvider = ({ children }: { children: ReactNode }) => {
    const value = useInferenceState();
    return <InferenceContext.Provider value={value}>{children}</InferenceContext.Provider>;
};

const useInference = () => {
    const value = useContext(InferenceContext);
    if (value === null) throw new Error('useInference must be used within an InferenceProvider');
    return value;
};

export const InferenceModelSummary = ({
    model,
    isPending,
    error,
}: {
    model: SchemaSessionModel;
    isPending: boolean;
    error: string | null;
}) => {
    const info = INFERENCE_BACKENDS[model.backend];
    const Logo = info?.logo;
    return (
        <div className={settingsClasses.modelSummary} data-error={error != null || undefined}>
            {Logo && (
                <span className={settingsClasses.backendLogo} aria-hidden='true'>
                    <Logo />
                </span>
            )}
            <PartRow
                label='Model'
                name={model.name}
                detail={`${info?.label ?? model.backend} · ${model.device}`}
                error={error}
                isPending={isPending}
            />
        </div>
    );
};

/** Settings stay visible while the selected mode determines the toolbar's primary action. */
export const InferenceMode = () => {
    const { task, setTask, executionMode, setExecutionMode, hz, setHz, threshold, setThreshold, isLocked } =
        useInference();
    return (
        <Flex direction='column' gap='size-150'>
            <div data-mode-interactive>
                <TextArea label='Task' width='100%' value={task} onChange={setTask} isReadOnly={isLocked} />
            </div>
            <div data-mode-interactive>
                <RadioGroup
                    label='Execution'
                    orientation='horizontal'
                    width='100%'
                    value={executionMode}
                    onChange={(value) => setExecutionMode(value as SchemaExecutionConfig['mode'])}
                    isDisabled={isLocked}
                >
                    {EXECUTION_MODES.map((item) => (
                        <Radio key={item.key} value={item.key}>
                            {item.label}
                        </Radio>
                    ))}
                </RadioGroup>
            </div>
            <div data-mode-interactive>
                <Disclosure isQuiet UNSAFE_className={settingsClasses.disclosure}>
                    <DisclosureTitle UNSAFE_className={settingsClasses.disclosureTitle}>
                        <Flex alignItems='center' gap='size-100'>
                            <Text UNSAFE_className={settingsClasses.disclosureText}>Execution settings</Text>
                            <Text UNSAFE_className={settingsClasses.hint}>
                                {hz} Hz · {Math.round(threshold * 100)}%
                            </Text>
                        </Flex>
                    </DisclosureTitle>
                    <DisclosurePanel
                        UNSAFE_className={settingsClasses.disclosurePanel}
                        UNSAFE_style={{ paddingBlock: 0, paddingInline: 0 }}
                    >
                        <Flex direction='column' gap='size-100'>
                            <Slider
                                label='Action rate'
                                width='100%'
                                minValue={5}
                                maxValue={100}
                                step={5}
                                value={hz}
                                onChange={setHz}
                                getValueLabel={(value) => `${value} Hz`}
                                isDisabled={isLocked}
                            />
                            <Flex direction='column' gap='size-50'>
                                <Slider
                                    label='Request threshold'
                                    width='100%'
                                    minValue={0}
                                    maxValue={1}
                                    step={0.05}
                                    value={threshold}
                                    onChange={setThreshold}
                                    formatOptions={{ style: 'percent' }}
                                    isDisabled={isLocked}
                                />
                                <Text UNSAFE_className={settingsClasses.guidance}>
                                    Request the next action chunk when this fraction remains.
                                </Text>
                            </Flex>
                        </Flex>
                    </DisclosurePanel>
                </Disclosure>
            </div>
        </Flex>
    );
};

export const InferenceToolbar = ({ backButton }: { backButton: ReactNode }) => {
    const { state } = useRuntimeSession();
    const { isRunning, isRecording, isPending, canRun, play, stop } = useInference();
    return (
        <div className={classes.toolbar} role='region' aria-label='Inference controls'>
            {backButton}
            <div className={classes.toolbarStatus} role='status'>
                <div className={classes.statusCopy}>
                    <strong>{isRunning ? 'Running model' : 'Model inference'}</strong>
                    <span className={classes.hint}>
                        {state?.model?.name ?? 'Choose a trained model in Session setup to begin.'}
                    </span>
                </div>
            </div>
            {isRunning ? (
                <Button variant='secondary' isPending={isPending} isDisabled={isRecording} onPress={stop}>
                    Stop model
                </Button>
            ) : (
                <Button variant='accent' isPending={isPending} isDisabled={!canRun} onPress={play}>
                    Run model
                </Button>
            )}
        </div>
    );
};
