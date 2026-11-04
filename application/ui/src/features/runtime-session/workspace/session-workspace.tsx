import { ReactNode, Suspense, useState } from 'react';

import { Button, Divider, Flex, Grid, Heading, Item, Loading, minmax, Picker, Text, View } from '@geti-ui/ui';

import { failureOf, SessionPartName, useRuntimeSession } from '../runtime-session-provider';
import { SessionViewer } from '../session-viewer';
import { InferenceMode } from './inference-mode';
import { PartAction, PartControlsProvider, usePartControls } from './part-controls';
import { EnvironmentRow, PartRow, PartsPanel } from './parts-panel';
import { DatasetRecordingSection, RecordButton, RecordMode } from './record-mode';

/**
 * One way of working with the session: the parts it needs, and its side bar panel, which
 * drives the follower with the mode's own control. Modes are combined deliberately, as a
 * mode of their own (human in the loop), not by showing several panels at once.
 */
interface WorkspaceMode {
    key: ModeKey;
    label: string;
    needs: SessionPartName[];
    panel: ReactNode;
}

export type ModeKey = 'record' | 'inference';

const MODES: WorkspaceMode[] = [
    { key: 'record', label: 'Record', needs: ['environment', 'dataset'], panel: <RecordMode /> },
    { key: 'inference', label: 'Inference', needs: ['environment', 'model'], panel: <InferenceMode /> },
];

const ARTICLE: Record<SessionPartName, string> = {
    environment: 'an environment',
    dataset: 'a dataset',
    model: 'a model',
};

/** What a mode needs that the session has not loaded, each with a way to load it. */
const MissingParts = ({ needs }: { needs: SessionPartName[] }) => {
    const { state } = useRuntimeSession();
    const { pendingPart, act } = usePartControls();
    const missing = needs.filter((part) => state?.[part] == null);

    if (missing.length === 0) {
        return null;
    }
    return (
        <View backgroundColor='gray-200' padding='size-150' borderRadius='regular'>
            <Flex direction='column' gap='size-100'>
                {missing.map((part) => (
                    <Flex key={part} alignItems='center' justifyContent='space-between' gap='size-100'>
                        <Text>Needs {ARTICLE[part]}.</Text>
                        <Button
                            variant='secondary'
                            isPending={pendingPart === part}
                            // The dataset and the model load onto an environment.
                            isDisabled={part !== 'environment' && state?.environment == null}
                            onPress={() => act(part, 'swap')}
                        >
                            Load {part}…
                        </Button>
                    </Flex>
                ))}
            </Flex>
        </View>
    );
};

const Sections = () => {
    const { state, statuses } = useRuntimeSession();
    const { pendingPart, act } = usePartControls();

    const environment = state?.environment;
    const dataset = state?.dataset;
    const model = state?.model;
    const environmentFailure = failureOf(statuses, 'environment');

    const loaded = (isLoaded: boolean): PartAction[] => (isLoaded ? ['reload', 'swap', 'unload'] : ['swap']);

    /** A dataset or model is kept without a working environment, but loading one needs it. */
    const partActions = (isLoaded: boolean): PartAction[] => {
        if (environment == null || environmentFailure !== null) {
            return isLoaded ? ['unload'] : [];
        }
        return loaded(isLoaded);
    };

    const row = (part: SessionPartName) => ({
        isPending: pendingPart === part,
        onAction: (action: PartAction) => act(part, action),
    });

    return (
        <Flex gap='size-200' direction='column'>
            <View isHidden>
                <RecordButton />
            </View>
            <View>
                <Heading level={4} margin={0}>
                    Environment
                </Heading>
                <EnvironmentRow
                    environment={environment}
                    statuses={statuses}
                    actions={loaded(environment != null)}
                    {...row('environment')}
                />
            </View>
            {/* Make this a Well like component */}
            <View>
                <Heading level={4} margin={0}>
                    Inference
                </Heading>
                <PartRow
                    label='Model'
                    name={model?.name}
                    detail={model ? `${model.backend} · ${model.device}` : undefined}
                    error={failureOf(statuses, 'model')}
                    isLoaded={model != null}
                    actions={partActions(model != null)}
                    {...row('model')}
                />
                <InferenceMode />
            </View>
            {/* Make this a Well like component */}
            <View>
                <Heading level={4} margin={0}>
                    Recording
                </Heading>
                <PartRow
                    label='Dataset'
                    name={dataset?.name}
                    detail={
                        dataset?.episodes_recorded ? `${dataset.episodes_recorded} recorded this session` : undefined
                    }
                    error={failureOf(statuses, 'dataset')}
                    isLoaded={dataset != null}
                    actions={partActions(dataset != null)}
                    {...row('dataset')}
                />
                <DatasetRecordingSection />
            </View>
        </Flex>
    );
};

/**
 * The session's robot and cameras in a dock, with a side bar: what is loaded, the mode, and the
 * selected mode's panel. Every mode is always there and says what it misses; entry points
 * (record, run model) only choose what is loaded first and which mode is selected.
 */
export const SessionWorkspace = ({ initialMode, cancelHref }: { initialMode: ModeKey; cancelHref: string }) => {
    const { send, state } = useRuntimeSession();
    const [modeKey, setModeKey] = useState(initialMode);
    const mode = MODES.find((candidate) => candidate.key === modeKey) ?? MODES[0];

    // TODO: change into three sections:
    // 1. Environment, allowing to switch environment, reload / unload if error
    // 2. Record, allowing to switch dataset, reload / unload if error
    //    Activate teleoperation
    // 3. Inference, allowing to switch model, reload / unload if error
    //    - inference settings

    const selectMode = (key: ModeKey) => {
        if (key === modeKey) {
            return;
        }
        // Each mode drives the follower its own way; hold it while switching.
        void send({ command: 'set_control', control: null });
        setModeKey(key);
    };

    return (
        <PartControlsProvider>
            <Grid
                columns={[minmax(0, '1fr'), 'size-4600']}
                gap='size-200'
                height='100%'
                minHeight={0}
                UNSAFE_style={{ gridTemplateRows: minmax(0, '1fr') }}
            >
                <SessionViewer cancelHref={cancelHref} />
                <View backgroundColor='gray-100' padding='size-200' overflow='auto' minHeight={0}>
                    <Flex direction='column' gap='size-200'>
                        <MissingParts needs={mode.needs} />

                        <Suspense fallback={<Loading mode='inline' />}>
                            <Sections />
                        </Suspense>
                    </Flex>
                </View>
            </Grid>
        </PartControlsProvider>
    );
};
