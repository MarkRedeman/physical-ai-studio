import { createContext, ReactNode, useContext, useEffect, useState } from 'react';

import { Button, Flex, Keyboard, Slider, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaSessionDataset } from '../../../api/openapi-spec';
import { formatKeyCombo, getEffectiveBindings } from '../../hotkeys/key-combo';
import { useHotkey } from '../../hotkeys/use-hotkey';
import { failureOf, useRuntimeSession } from '../runtime-session-provider';
import { CurrentTaskPicker } from './current-task-picker';
import { usePartControls } from './part-controls';
import { useSessionMode } from './session-mode';

import classes from './record-mode.module.css';

const RecordingTimer = () => {
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
        const started = Date.now();
        const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
        return () => window.clearInterval(timer);
    }, []);

    const minutes = Math.floor(seconds / 60)
        .toString()
        .padStart(2, '0');
    const remaining = (seconds % 60).toString().padStart(2, '0');

    return (
        <span className={classes.timer} aria-label='Elapsed recording time'>
            {minutes}:{remaining}
        </span>
    );
};

/** Shared by the task editor, toolbar and a single set of keyboard shortcuts. */
const useRecordingState = () => {
    const { state, statuses, isLoading } = useRuntimeSession();
    const { mode, isPending, run } = useSessionMode();
    const { pendingPart } = usePartControls();
    const dataset = state?.dataset;
    const { data: savedDataset, isPending: isTaskLoading } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        { params: { path: { dataset_id: dataset?.dataset_id ?? '' } } },
        { enabled: dataset != null }
    );
    const [task, setTask] = useState('');
    const [savedTasks, setSavedTasks] = useState<Record<string, string[]>>({});
    const defaultTask = savedDataset?.default_task ?? '';

    useEffect(() => setTask(defaultTask), [defaultTask, dataset?.dataset_id]);

    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const bindings = getEffectiveBindings(settings.hotkeys.bindings);
    const isRecording = dataset?.is_recording ?? false;
    const isReady =
        state?.environment != null &&
        failureOf(statuses, 'environment') === null &&
        dataset != null &&
        failureOf(statuses, 'dataset') === null;
    const canRecord = mode === 'record' && isReady && !isPending && !isLoading && pendingPart === null;
    const canStart = canRecord && !isRecording && !isTaskLoading && task.trim() !== '';

    const start = () => {
        if (canStart) void run({ command: 'start_episode', task: task.trim() });
    };
    // Saving/discarding stays available if a device fails during an episode.
    const save = () => {
        if (isRecording && !isPending) {
            void run({ command: 'save_episode' }).then((ack) => {
                if (ack?.ok && dataset != null && task.trim() !== '') {
                    // The episodes endpoint does not include this session until the dataset is closed.
                    setSavedTasks((previous) => ({
                        ...previous,
                        [dataset.dataset_id]: [
                            task.trim(),
                            ...(previous[dataset.dataset_id] ?? []).filter((saved) => saved !== task.trim()),
                        ],
                    }));
                }
            });
        }
    };
    const discard = () => {
        if (isRecording && !isPending) void run({ command: 'discard_episode' });
    };

    useHotkey(bindings['recording.start_episode'], start, canStart);
    useHotkey(bindings['recording.accept_episode'], save, isRecording && !isPending);
    useHotkey(bindings['recording.discard_episode'], discard, isRecording && !isPending);

    return {
        task,
        setTask,
        isRecording,
        isPending,
        canStart,
        bindings,
        start,
        save,
        discard,
        run,
        recentTasks: savedTasks[dataset?.dataset_id ?? ''] ?? [],
    };
};

const RecordingContext = createContext<ReturnType<typeof useRecordingState> | null>(null);

export const RecordingProvider = ({ children }: { children: ReactNode }) => {
    const recording = useRecordingState();
    return <RecordingContext.Provider value={recording}>{children}</RecordingContext.Provider>;
};

const useRecording = () => {
    const recording = useContext(RecordingContext);
    if (recording === null) throw new Error('useRecording must be used within a RecordingProvider');
    return recording;
};

/** The rate is editable only until the dataset has its first episode. */
const RecordingRate = ({ dataset }: { dataset: SchemaSessionDataset }) => {
    const { statuses, isLoading } = useRuntimeSession();
    const { run, isPending } = useRecording();
    const { pendingPart } = usePartControls();
    const [fps, setFps] = useState(dataset.fps);
    useEffect(() => setFps(dataset.fps), [dataset.fps]);

    const reload = async (value: number) => {
        if (value === dataset.fps) return;
        const ack = await run({ command: 'load_dataset', dataset_id: dataset.dataset_id, fps: value });
        if (!ack?.ok) setFps(dataset.fps);
    };

    return (
        <Flex direction='column' gap='size-50'>
            <Slider
                label='Recording rate'
                value={fps}
                onChange={setFps}
                onChangeEnd={(value) => void reload(value)}
                minValue={10}
                maxValue={100}
                step={5}
                getValueLabel={(value) => `${value} fps`}
                width='100%'
                isDisabled={
                    dataset.is_recording ||
                    isPending ||
                    isLoading ||
                    pendingPart !== null ||
                    failureOf(statuses, 'environment') !== null
                }
            />
            <Text UNSAFE_className={classes.hint}>Locks after your first saved episode.</Text>
        </Flex>
    );
};

export const RecordingToolbar = ({ backButton }: { backButton: ReactNode }) => {
    const { state, statuses, isLoading } = useRuntimeSession();
    const { task, isRecording, isPending, canStart, bindings, start, save, discard } = useRecording();
    const getHint = () => {
        if (isLoading) return 'Preparing the session…';
        if (state?.environment == null || failureOf(statuses, 'environment') !== null) {
            return 'Load a working environment to record episodes';
        }
        if (state.dataset == null) return 'Load a dataset to record episodes';
        if (failureOf(statuses, 'dataset') !== null) return 'Check the dataset in Session setup to continue';
        if (task.trim() === '') return 'Add a task to start recording';
        return 'Start an episode to capture the current task.';
    };

    return (
        <div className={classes.toolbar} aria-label='Episode controls' role='region'>
            {backButton}
            <div className={classes.toolbarStatus} role='status'>
                {isRecording && <span className={classes.recordingDot} aria-hidden='true' />}
                <div className={classes.statusCopy}>
                    <strong>
                        {isRecording ? 'Recording episode' : canStart ? 'Ready to record' : 'Record an episode'}
                    </strong>
                    <span className={classes.hint}>
                        {isRecording ? 'Save this episode or discard it to try again.' : getHint()}
                    </span>
                </div>
                {isRecording && <RecordingTimer />}
            </div>
            <Flex gap='size-100' flexShrink={0} wrap>
                {isRecording ? (
                    <>
                        <Button variant='secondary' isDisabled={isPending} onPress={discard}>
                            <Text>Discard</Text>
                            <Keyboard UNSAFE_className={classes.hotkey}>
                                {formatKeyCombo(bindings['recording.discard_episode'])}
                            </Keyboard>
                        </Button>
                        <Button variant='accent' isPending={isPending} onPress={save}>
                            <Text>Save episode</Text>
                            <Keyboard UNSAFE_className={classes.hotkey}>
                                {formatKeyCombo(bindings['recording.accept_episode'])}
                            </Keyboard>
                        </Button>
                    </>
                ) : (
                    <Button variant='accent' isDisabled={!canStart} isPending={isPending} onPress={start}>
                        <Text>Start episode</Text>
                        <Keyboard UNSAFE_className={classes.hotkey}>
                            {formatKeyCombo(bindings['recording.start_episode'])}
                        </Keyboard>
                    </Button>
                )}
            </Flex>
        </div>
    );
};

export const DatasetRecordingSection = () => {
    const { state } = useRuntimeSession();
    const { task, setTask, isRecording, isPending, recentTasks } = useRecording();
    const dataset = state?.dataset;
    if (dataset == null) return null;
    const isRateFixed = dataset.fps_fixed || dataset.episodes_recorded > 0;

    return (
        <Flex direction='column' gap='size-150'>
            {!isRateFixed && <RecordingRate key={dataset.dataset_id} dataset={dataset} />}
            <CurrentTaskPicker
                dataset_id={dataset.dataset_id}
                task={task}
                onChange={setTask}
                recentTasks={recentTasks}
                isReadOnly={isRecording || isPending}
            />
        </Flex>
    );
};
