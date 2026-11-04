import { ReactNode, Suspense } from 'react';

import { Flex, Heading, Loading, StatusLight, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { useProjectId } from '../../projects/use-project';
import { TeleopSwitch } from '../cells/robot-cell';
import { failureOf, SessionPartName, useRuntimeSession } from '../runtime-session-provider';
import { SessionViewer } from '../session-viewer';
import { InferenceMode } from './inference-mode';
import { PartAction, PartControlsProvider, usePartControls } from './part-controls';
import { EnvironmentRow, PartRow } from './parts-panel';
import { DatasetRecordingSection, RecordingProvider, RecordingToolbar } from './record-mode';

import classes from './session-workspace.module.css';

export type ModeKey = 'record' | 'inference';

const WorkspaceSection = ({
    title,
    children,
    isActive = false,
}: {
    title: string;
    children: ReactNode;
    isActive?: boolean;
}) => (
    <section className={classes.section} data-active={isActive || undefined} aria-label={title}>
        <div className={classes.sectionHeading}>
            <Heading level={3} margin={0}>
                {title}
            </Heading>
            {isActive && <StatusLight variant='positive'>Active</StatusLight>}
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
    const environmentFailure = failureOf(statuses, 'environment');
    const isRecording = dataset?.is_recording ?? false;
    const isTeleop = state?.control?.type === 'teleop';

    const loaded = (isLoaded: boolean): PartAction[] => (isLoaded ? ['reload', 'swap', 'unload'] : ['swap']);
    const partActions = (isLoaded: boolean): PartAction[] => {
        if (environment == null || environmentFailure !== null) return isLoaded ? ['unload'] : [];
        return loaded(isLoaded);
    };
    const row = (part: SessionPartName) => ({
        isPending: pendingPart === part,
        onAction: (action: PartAction) => act(part, action),
        // Loading or replacing resources during an episode can discard it.
        isDisabled: isRecording || pendingPart !== null,
    });

    const recording = (
        <WorkspaceSection title='Recording' isActive={isRecording}>
            <PartRow
                label='Dataset'
                name={dataset?.name}
                detail={dataset ? `${dataset.episodes_recorded} saved this session` : undefined}
                error={failureOf(statuses, 'dataset')}
                isLoaded={dataset != null}
                actions={partActions(dataset != null)}
                {...row('dataset')}
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
        model == null && models !== undefined && !hasRunnableModels ? (
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
            <WorkspaceSection title='Inference' isActive={state?.control?.type === 'model'}>
                <PartRow
                    label='Model'
                    name={model?.name}
                    detail={model ? `${model.backend} · ${model.device}` : undefined}
                    error={failureOf(statuses, 'model')}
                    isLoaded={model != null}
                    actions={partActions(model != null)}
                    {...row('model')}
                />
                {model != null ? (
                    <InferenceMode />
                ) : (
                    <Text UNSAFE_className={classes.hint}>Load a trained model to run it on this environment.</Text>
                )}
            </WorkspaceSection>
        );

    return (
        <>
            <WorkspaceSection title='Environment'>
                <EnvironmentRow
                    environment={environment}
                    statuses={statuses}
                    actions={loaded(environment != null)}
                    {...row('environment')}
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
export const SessionWorkspace = ({ initialMode, cancelHref }: { initialMode: ModeKey; cancelHref: string }) => (
    <PartControlsProvider>
        <Suspense fallback={<Loading mode='inline' />}>
            <RecordingProvider>
                <div className={classes.workspace}>
                    <main className={classes.liveWorkspace} aria-label='Live session'>
                        <RecordingToolbar />
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
