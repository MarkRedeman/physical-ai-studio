import { ReactNode, Suspense, useEffect, useId, useState } from 'react';

import { ActionButton, Flex, Heading, Loading, Text } from '@geti-ui/ui';
import { ChevronDownLight, ChevronLeft, ChevronRightSmallLight, Play } from '@geti-ui/ui/icons';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { useProjectId } from '../../projects/use-project';
import { TeleopSwitch } from '../cells/robot-cell';
import { failureOf, SessionPartName, useRuntimeSession } from '../runtime-session-provider';
import { SessionViewer } from '../session-viewer';
import { InferenceMode } from './inference-mode';
import { PartControlsProvider, usePartControls } from './part-controls';
import { EnvironmentRow, PartActions, PartRow } from './parts-panel';
import { DatasetRecordingSection, RecordingProvider, RecordingToolbar } from './record-mode';

import classes from './session-workspace.module.css';

export type ModeKey = 'record' | 'inference';

const BackButton = ({ href }: { href: string }) => {
    const navigate = useNavigate();
    const { state } = useRuntimeSession();
    return (
        <ActionButton
            isQuiet
            onPress={() => navigate(href)}
            aria-label='Back'
            isDisabled={state?.dataset?.is_recording}
        >
            <ChevronLeft />
            <Text>Back</Text>
        </ActionButton>
    );
};

const WorkspaceSection = ({
    title,
    children,
    isActive = false,
    actions,
    hasError = false,
}: {
    title: string;
    children: ReactNode;
    isActive?: boolean;
    actions?: ReactNode;
    hasError?: boolean;
}) => (
    <section
        className={classes.section}
        data-active={isActive || undefined}
        data-error={hasError || undefined}
        aria-label={title}
    >
        <div className={classes.sectionHeading}>
            <Heading level={3} margin={0}>
                {title}
            </Heading>
            <Flex alignItems='center' gap='size-100'>
                {isActive && <Text UNSAFE_className={classes.activeLabel}>Active</Text>}
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
    const { pendingPart, act } = usePartControls();
    const environment = state?.environment;
    const dataset = state?.dataset;
    const model = state?.model;
    const modelFailure = failureOf(statuses, 'model');
    const environmentFailure = failureOf(statuses, 'environment');
    const isRecording = dataset?.is_recording ?? false;
    const isTeleop = state?.control?.type === 'teleop';
    const savedEpisodes = dataset?.episodes_recorded ?? 0;
    const recordingDetail =
        savedEpisodes === 0
            ? 'No episodes saved this session yet'
            : `${savedEpisodes} ${savedEpisodes === 1 ? 'episode' : 'episodes'} saved this session`;
    const devices = Object.values(statuses).filter((status) => status.component.startsWith('environment/'));
    const deviceFailures = devices.filter((status) => status.state === 'failed');
    const hasDeviceFailure = deviceFailures.length > 0;
    const failureKey = deviceFailures.map(({ component, error }) => `${component}:${error}`).join('|');
    const [showDeviceDetails, setShowDeviceDetails] = useState(false);
    const deviceDetailsId = useId();
    const canShowDevices = environment != null;
    useEffect(() => {
        if (environmentFailure !== null || failureKey !== '') setShowDeviceDetails(true);
    }, [environmentFailure, failureKey]);

    const actions = (part: SessionPartName) => (
        <PartActions
            part={part}
            isLoaded={state?.[part] != null}
            isPending={pendingPart === part}
            isDisabled={isRecording || pendingPart !== null}
            canChoose={part === 'environment' || (environment != null && environmentFailure === null)}
            canRetry={failureOf(statuses, part) !== null || (part === 'environment' && hasDeviceFailure)}
            onAction={(action) => act(part, action)}
        />
    );

    const recording = (
        <WorkspaceSection
            title='Recording'
            isActive={isRecording}
            actions={actions('dataset')}
            hasError={failureOf(statuses, 'dataset') !== null}
        >
            <PartRow
                label='Dataset'
                name={dataset?.name}
                detail={dataset ? recordingDetail : undefined}
                error={failureOf(statuses, 'dataset')}
                isPending={pendingPart === 'dataset'}
            />
            {dataset != null ? (
                <DatasetRecordingSection />
            ) : (
                <Text UNSAFE_className={classes.hint}>Load a dataset to capture demonstrations or model runs.</Text>
            )}
        </WorkspaceSection>
    );
    const hasRunnableModels = models?.some((candidate) => candidate.available_backends.length > 0);
    const inference =
        model == null &&
        models !== undefined &&
        !hasRunnableModels &&
        failureOf(statuses, 'model') === null &&
        pendingPart !== 'model' ? (
            <section className={classes.emptySection} aria-label='Inference'>
                <Heading level={3} margin={0}>
                    Inference
                </Heading>
                <Text UNSAFE_className={classes.hint}>
                    {models.length === 0
                        ? 'Start with demonstrations. Once you train a model, you can run it here.'
                        : 'Export a trained model to run it here.'}
                </Text>
            </section>
        ) : (
            <WorkspaceSection
                title='Inference'
                isActive={state?.control?.type === 'model'}
                actions={model != null ? actions('model') : undefined}
                hasError={modelFailure !== null}
            >
                {model != null ? (
                    <>
                        <PartRow
                            label='Model'
                            name={model?.name}
                            detail={model ? `${model.backend} · ${model.device}` : undefined}
                            error={failureOf(statuses, 'model')}
                            isPending={pendingPart === 'model'}
                        />
                        <InferenceMode />
                    </>
                ) : (
                    <>
                        {modelFailure !== null && <Text UNSAFE_className={classes.error}>{modelFailure}</Text>}
                        <button
                            type='button'
                            className={classes.modelWell}
                            onClick={() => act('model', 'swap')}
                            disabled={
                                isRecording ||
                                pendingPart !== null ||
                                !hasRunnableModels ||
                                environment == null ||
                                environmentFailure !== null
                            }
                            aria-label='Choose a model for inference'
                        >
                            <span className={classes.modelWellIcon} aria-hidden='true'>
                                <Play />
                            </span>
                            <strong>{pendingPart === 'model' ? 'Preparing model…' : 'Choose a model'}</strong>
                            <span>
                                {environment == null || environmentFailure !== null
                                    ? 'Connect an environment to run a model.'
                                    : 'Run a trained policy on your environment.'}
                            </span>
                        </button>
                    </>
                )}
            </WorkspaceSection>
        );

    return (
        <>
            <WorkspaceSection
                title='Environment'
                actions={
                    <>
                        {canShowDevices && (
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
                hasError={environmentFailure !== null || hasDeviceFailure}
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
                        <TeleopSwitch isDisabled={pendingPart !== null} />
                        <Text UNSAFE_className={classes.hint}>
                            {isTeleop
                                ? 'The follower tracks your leader robot.'
                                : 'Enable to guide the follower with your leader robot.'}
                        </Text>
                    </div>
                )}
            </WorkspaceSection>
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
        </>
    );
};

/** One workspace for recording demonstrations and running models, without switching modes. */
export const SessionWorkspace = ({
    initialMode,
    cancelHref,
    backButton,
}: {
    initialMode: ModeKey;
    cancelHref: string;
    backButton?: ReactNode;
}) => (
    <PartControlsProvider>
        <Suspense fallback={<Loading mode='inline' />}>
            <RecordingProvider>
                <div className={classes.workspace}>
                    <main className={classes.liveWorkspace} aria-label='Live session'>
                        <RecordingToolbar backButton={backButton ?? <BackButton href={cancelHref} />} />
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
                                Configure your environment, recording and model.
                            </Text>
                        </div>
                        <Sections initialMode={initialMode} />
                    </aside>
                </div>
            </RecordingProvider>
        </Suspense>
    </PartControlsProvider>
);
