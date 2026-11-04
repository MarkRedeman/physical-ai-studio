import { useEffect, useState } from 'react';

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
    toast,
} from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaExecutionConfig, SchemaModelControlConfig } from '../../../api/openapi-spec';
import { ERROR_TOAST, failureOf, useRuntimeSession } from '../runtime-session-provider';
import { usePartControls } from './part-controls';

const EXECUTION_MODES: { key: SchemaExecutionConfig['mode']; label: string }[] = [
    { key: 'sync', label: 'Synchronous' },
    { key: 'async', label: 'Asynchronous' },
];

/** The default task of the dataset the loaded model was trained on, or none. */
const useDefaultTask = (model_id: string | undefined) => {
    const { data: model } = $api.useQuery(
        'get',
        '/api/models/{model_id}',
        { params: { path: { model_id: model_id ?? '' } } },
        { enabled: model_id !== undefined }
    );
    const dataset_id = model?.model.dataset_id ?? undefined;
    const { data: dataset } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        { params: { path: { dataset_id: dataset_id ?? '' } } },
        { enabled: dataset_id !== undefined }
    );
    return dataset?.default_task ?? '';
};

/** Inference mode's side bar: the task, the action rate and how chunks are requested, then play / stop. */
export const InferenceMode = () => {
    const { state, statuses, send, isLoading } = useRuntimeSession();
    const { pendingPart } = usePartControls();
    const defaultTask = useDefaultTask(state?.model?.model_id);
    const [task, setTask] = useState(defaultTask);
    const [mode, setMode] = useState<SchemaExecutionConfig['mode']>('sync');
    const [threshold, setThreshold] = useState(0.5);
    // shortcut: set by hand, upgrade to the model's training fps once exports record it.
    const [hz, setHz] = useState(30);
    const [isPending, setIsPending] = useState(false);

    useEffect(() => setTask(defaultTask), [defaultTask]);

    const isRunning = state?.control?.type === 'model';
    const isRecording = state?.dataset?.is_recording ?? false;
    const isLocked = isRunning || isRecording || isPending || isLoading || pendingPart !== null;
    const canRun =
        state?.environment != null &&
        failureOf(statuses, 'environment') === null &&
        state.model != null &&
        failureOf(statuses, 'model') === null &&
        !isLocked;

    const setControl = async (control: SchemaModelControlConfig | null) => {
        setIsPending(true);
        try {
            const ack = await send({ command: 'set_control', control });
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? 'Could not run the model.', ERROR_TOAST);
            }
        } finally {
            setIsPending(false);
        }
    };

    const play = () =>
        void setControl({
            type: 'model',
            task,
            hz,
            execution: { mode, request_threshold: threshold },
            // The follower eases onto the model's actions, which may start far from where it is.
            grace_s: 3,
        });

    return (
        <Flex direction='column' gap='size-200'>
            <TextArea label='Task' width='100%' value={task} onChange={setTask} isReadOnly={isLocked} />

            <RadioGroup
                label='Execution'
                width='100%'
                value={mode}
                onChange={(value) => setMode(value as SchemaExecutionConfig['mode'])}
                isDisabled={isLocked}
            >
                {EXECUTION_MODES.map((item) => (
                    <Radio key={item.key} value={item.key}>
                        {item.label}
                    </Radio>
                ))}
            </RadioGroup>

            <Disclosure isQuiet>
                <DisclosureTitle>Execution settings</DisclosureTitle>
                <DisclosurePanel>
                    <Flex direction='column' gap='size-150'>
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
                        {mode === 'async' && (
                            <>
                                <Slider
                                    label='Request the next chunk at'
                                    width='100%'
                                    minValue={0}
                                    maxValue={1}
                                    step={0.05}
                                    value={threshold}
                                    onChange={setThreshold}
                                    formatOptions={{ style: 'percent' }}
                                    isDisabled={isLocked}
                                />
                                <Text>
                                    Request another action chunk when this fraction of the current chunk remains.
                                </Text>
                            </>
                        )}
                    </Flex>
                </DisclosurePanel>
            </Disclosure>
            {isRecording && <Text>Save or discard the episode before changing robot control.</Text>}
            {isRunning ? (
                <Button
                    variant='secondary'
                    isDisabled={isRecording}
                    isPending={isPending}
                    onPress={() => void setControl(null)}
                >
                    Stop model
                </Button>
            ) : (
                <Button isPending={isPending} isDisabled={!canRun} onPress={play}>
                    Run model
                </Button>
            )}
        </Flex>
    );
};
