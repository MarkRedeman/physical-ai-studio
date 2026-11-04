import { ReactNode, Suspense, useEffect, useId, useState } from 'react';

import { ActionButton, Flex, Heading, Loading, Radio, RadioGroup, Text } from '@geti-ui/ui';
import { ChevronDownLight, ChevronLeft, ChevronRightSmallLight } from '@geti-ui/ui/icons';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { useProjectId } from '../../projects/use-project';
import { TeleopSwitch } from '../cells/robot-cell';
import { failureOf, SessionPartName, useRuntimeSession } from '../runtime-session-provider';
import { SessionViewer } from '../session-viewer';
import { InferenceMode, InferenceProvider, InferenceToolbar } from './inference-mode';
import { PartControlsProvider, usePartControls } from './part-controls';
import { EnvironmentRow, PartActions, PartRow } from './parts-panel';
import { DatasetRecordingSection, RecordingProvider, RecordingToolbar } from './record-mode';
import { RecordingSummary } from './recording-summary';
import { ModeKey, SessionModeProvider, useSessionMode } from './session-mode';

import classes from './session-workspace.module.css';

export type { ModeKey } from './session-mode';

interface SessionWorkspaceProps {
    initialMode: ModeKey;
    cancelHref: string;
    backButton?: ReactNode;
}

const BackButton = ({ href }: { href: string }) => {
    const navigate = useNavigate();
    const { state } = useRuntimeSession();
    const { isPending } = useSessionMode();
    return (
        <ActionButton
            isQuiet
            onPress={() => navigate(href)}
            aria-label='Back'
            isDisabled={isPending || state?.dataset?.is_recording}
        >
            <ChevronLeft />
            <Text>Back</Text>
        </ActionButton>
    );
};

const WorkspaceSection = ({
    title,
    children,
    isSelected = false,
    modeKey,
    activity,
    isQuiet = false,
    actions,
    hasError = false,
}: {
    title: string;
    children: ReactNode;
    isSelected?: boolean;
    modeKey?: ModeKey;
    activity?: string;
    isQuiet?: boolean;
    actions?: ReactNode;
    hasError?: boolean;
}) => (
    <section
        className={classes.section}
        data-active={isSelected || undefined}
        data-empty={isQuiet || undefined}
        data-error={hasError || undefined}
        aria-label={title}
    >
        <div className={classes.sectionHeading}>
            <Heading level={3} margin={0}>
                {modeKey ? (
                    <Radio value={modeKey} UNSAFE_className={classes.modeRadio}>
                        <Text UNSAFE_className={classes.modeLabel}>{title}</Text>
                    </Radio>
                ) : (
                    title
                )}
            </Heading>
            <Flex alignItems='center' gap='size-100'>
                {activity && <Text UNSAFE_className={classes.activeLabel}>{activity}</Text>}
                {actions}
            </Flex>
        </div>
        <Flex direction='column' gap='size-200'>
            {children}
        </Flex>
    </section>
);

const Sections = ({ initialMode }: { initialMode: ModeKey }) => {
    const { project_id } = useProjectId();
    const { data: models } = $api.useQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id } },
    });
    const { state, statuses } = useRuntimeSession();
    const { mode, selectMode, canSwitch, isPending } = useSessionMode();
    const { pendingPart, act } = usePartControls();
    const environment = state?.environment;
    const dataset = state?.dataset;
    const model = state?.model;
    const modelFailure = failureOf(statuses, 'model');
    const environmentFailure = failureOf(statuses, 'environment');
    const isRecording = dataset?.is_recording ?? false;
    const isTeleop = state?.control?.type === 'teleop';
    const devices = Object.values(statuses).filter((status) => status.component.startsWith('environment/'));
    const deviceFailures = devices.filter((status) => status.state === 'failed');
    const hasDeviceFailure = deviceFailures.length > 0;
    const failureKey = deviceFailures.map(({ component, error }) => `${component}:${error}`).join('|');
    const [showDeviceDetails, setShowDeviceDetails] = useState(false);
    const deviceDetailsId = useId();
    useEffect(() => {
        if (environmentFailure !== null || failureKey !== '') setShowDeviceDetails(true);
    }, [environmentFailure, failureKey]);

    const actions = (part: SessionPartName) => (
        <PartActions
            part={part}
            isLoaded={state?.[part] != null}
            isPending={pendingPart === part}
            isDisabled={isRecording || isPending || pendingPart !== null}
            canChoose={part === 'environment' || (environment != null && environmentFailure === null)}
            canRetry={failureOf(statuses, part) !== null || (part === 'environment' && hasDeviceFailure)}
            onAction={(action) => act(part, action)}
        />
    );

    const recording = (
        <WorkspaceSection
            title='Recording'
            modeKey='record'
            isSelected={mode === 'record'}
            activity={isRecording ? 'Recording' : undefined}
            actions={actions('dataset')}
            hasError={failureOf(statuses, 'dataset') !== null}
        >
            <RecordingSummary />
            <PartRow
                label='Dataset'
                name={dataset?.name}
                detail={
                    dataset && (dataset.fps_fixed || dataset.episodes_recorded > 0)
                        ? `Recording at ${dataset.fps} fps · fixed for this dataset`
                        : undefined
                }
                error={failureOf(statuses, 'dataset')}
                isPending={pendingPart === 'dataset'}
            />
            {dataset != null ? (
                <DatasetRecordingSection />
            ) : (
                <Text UNSAFE_className={classes.hint}>Load a dataset to capture demonstrations.</Text>
            )}
        </WorkspaceSection>
    );
    const hasRunnableModels = models?.some((candidate) => candidate.available_backends.length > 0);
    const isEmptyInference =
        model == null && models !== undefined && !hasRunnableModels && modelFailure === null && pendingPart !== 'model';
    const inference = (
        <WorkspaceSection
            title='Inference'
            modeKey='inference'
            isSelected={mode === 'inference'}
            activity={state?.control?.type === 'model' ? 'Running' : undefined}
            isQuiet={isEmptyInference}
            actions={model != null ? actions('model') : undefined}
            hasError={modelFailure !== null}
        >
            {isEmptyInference ? (
                <Text UNSAFE_className={classes.hint}>
                    {models?.length === 0
                        ? 'Start with demonstrations. Once you train a model, you can run it here.'
                        : 'Export a trained model to run it here.'}
                </Text>
            ) : model != null ? (
                <>
                    <PartRow
                        label='Model'
                        name={model.name}
                        detail={`${model.backend} · ${model.device}`}
                        error={modelFailure}
                        isPending={pendingPart === 'model'}
                    />
                    <InferenceMode />
                </>
            ) : (
                <>
                    {modelFailure !== null && <Text UNSAFE_className={classes.error}>{modelFailure}</Text>}
                    <button
                        type='button'
                        className={classes.modelPicker}
                        onClick={() => act('model', 'swap')}
                        disabled={
                            isRecording ||
                            isPending ||
                            pendingPart !== null ||
                            !hasRunnableModels ||
                            environment == null ||
                            environmentFailure !== null
                        }
                        aria-label='Choose a model for inference'
                    >
                        <span className={classes.modelPickerCopy}>
                            <strong>{pendingPart === 'model' ? 'Preparing model…' : 'Select a trained model'}</strong>
                            <span>
                                {environment == null || environmentFailure !== null
                                    ? 'Connect an environment first'
                                    : 'Browse your available models'}
                            </span>
                        </span>
                        <ChevronRightSmallLight aria-hidden='true' />
                    </button>
                    <Text UNSAFE_className={classes.hint}>The selected model will control the follower robot.</Text>
                </>
            )}
        </WorkspaceSection>
    );

    return (
        <>
            <WorkspaceSection
                title='Environment'
                hasError={environmentFailure !== null || hasDeviceFailure}
                actions={
                    <>
                        {environment != null && (
                            <ActionButton
                                isQuiet
                                aria-label={showDeviceDetails ? 'Hide device details' : 'Show device details'}
                                aria-expanded={showDeviceDetails}
                                aria-controls={deviceDetailsId}
                                onPress={() => setShowDeviceDetails(!showDeviceDetails)}
                            >
                                {showDeviceDetails ? <ChevronDownLight /> : <ChevronRightSmallLight />}
                            </ActionButton>
                        )}
                        {actions('environment')}
                    </>
                }
            >
                <EnvironmentRow
                    environment={environment}
                    statuses={statuses}
                    isPending={pendingPart === 'environment'}
                    showDetails={showDeviceDetails}
                    detailsId={deviceDetailsId}
                    control={state?.control}
                />
                {environment?.leader_id != null && (
                    <div className={classes.teleop} data-active={isTeleop || undefined}>
                        <TeleopSwitch isDisabled={mode !== 'record' || isPending || pendingPart !== null} />
                        <Text UNSAFE_className={classes.hint}>
                            {mode !== 'record'
                                ? 'Select Recording to guide the follower with your leader robot.'
                                : isTeleop
                                  ? 'The follower tracks your leader robot.'
                                  : 'Enable to guide the follower with your leader robot.'}
                        </Text>
                    </div>
                )}
            </WorkspaceSection>
            <RadioGroup
                aria-label='Session mode'
                value={mode}
                isEmphasized
                isDisabled={!canSwitch}
                onChange={(value) => {
                    if (value === 'record' || value === 'inference') void selectMode(value);
                }}
                width='100%'
                UNSAFE_className={classes.modeGroup}
            >
                <Flex direction='column' gap='size-150'>
                    {initialMode === 'record' ? (
                        <>
                            {recording}
                            {inference}
                        </>
                    ) : (
                        <>
                            {inference}
                            {recording}
                        </>
                    )}
                </Flex>
            </RadioGroup>
        </>
    );
};

const WorkspaceContent = ({ initialMode, cancelHref, backButton }: SessionWorkspaceProps) => {
    const { mode } = useSessionMode();
    const { state } = useRuntimeSession();
    const back = backButton ?? <BackButton href={cancelHref} />;
    return (
        <div className={classes.workspace}>
            <main className={classes.liveWorkspace} aria-label='Live session'>
                {mode === 'record' ? <RecordingToolbar backButton={back} /> : <InferenceToolbar backButton={back} />}
                <div className={classes.viewer}>
                    <SessionViewer cancelHref={cancelHref} />
                </div>
            </main>
            <aside className={classes.sidebar} aria-label='Session setup'>
                <div className={classes.sidebarHeading}>
                    <Heading level={2} margin={0}>
                        Session setup
                    </Heading>
                    <Text UNSAFE_className={classes.hint}>
                        {state?.dataset?.is_recording
                            ? 'Save or discard this episode before switching modes.'
                            : 'Select a mode to record demonstrations or run a model.'}
                    </Text>
                </div>
                <Sections initialMode={initialMode} />
            </aside>
        </div>
    );
};

/** Both mode cards stay visible; selection changes the primary action, not the layout. */
export const SessionWorkspace = ({ initialMode, cancelHref, backButton }: SessionWorkspaceProps) => (
    <PartControlsProvider>
        <SessionModeProvider initialMode={initialMode}>
            <Suspense fallback={<Loading mode='inline' />}>
                <RecordingProvider>
                    <InferenceProvider>
                        <WorkspaceContent initialMode={initialMode} cancelHref={cancelHref} backButton={backButton} />
                    </InferenceProvider>
                </RecordingProvider>
            </Suspense>
        </SessionModeProvider>
    </PartControlsProvider>
);
