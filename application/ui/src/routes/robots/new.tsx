import { Suspense, useCallback, useRef, useState } from 'react';

import { Button, Divider, Flex, Grid, Heading, Loading, View } from '@geti/ui';

import { SchemaRobotType } from '../../api/openapi-spec';
import { RobotForm } from '../../features/robots/robot-form/form';
import { RobotFormProvider, useRobotForm } from '../../features/robots/robot-form/provider';
import { SubmitNewRobotButton } from '../../features/robots/robot-form/submit-new-robot-button';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { SetupRobotViewer } from '../../features/robots/setup-wizard/shared/setup-robot-viewer';
import { Stepper } from '../../features/robots/setup-wizard/shared/stepper';
import { JointHighlight } from '../../features/robots/setup-wizard/shared/use-joint-highlight';
import { SO101StepBody, SO101ViewerEffects } from '../../features/robots/setup-wizard/so101/setup-wizard';
import {
    SetupWizardProvider,
    STEP_LABELS as SO101_STEP_LABELS,
    WIZARD_STEPS as SO101_WIZARD_STEPS,
    useSetupActions,
    useSetupState,
    WizardStep,
} from '../../features/robots/setup-wizard/so101/wizard-provider';
import { TrossenStepBody } from '../../features/robots/setup-wizard/trossen/setup-wizard';
import {
    TROSSEN_STEP_LABELS,
    TROSSEN_WIZARD_STEPS,
    TrossenSetupWizardProvider,
    TrossenWizardStep,
    useTrossenSetupActions,
    useTrossenSetupState,
} from '../../features/robots/setup-wizard/trossen/wizard-provider';

import classes from '../../features/robots/setup-wizard/shared/setup-wizard.module.scss';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const CenteredLoading = () => (
    <Flex width='100%' height='100%' alignItems='center' justifyContent='center'>
        <Loading mode='inline' />
    </Flex>
);

/** Derive the robot family from the selected type string. */
function robotFamily(type: string | null): 'so101' | 'trossen' | null {
    if (!type) return null;
    const lower = type.toLowerCase();
    if (lower.startsWith('so101')) return 'so101';
    if (lower.includes('trossen')) return 'trossen';
    return null;
}

// ---------------------------------------------------------------------------
// Unified stepper steps — 'robot_info' + wizard-specific steps
// ---------------------------------------------------------------------------

const ROBOT_INFO_STEP = 'robot_info' as const;
const ROBOT_INFO_LABEL = 'Robot Info';

function getUnifiedSteps(family: 'so101' | 'trossen' | null): string[] {
    if (family === 'so101') return [ROBOT_INFO_STEP, ...SO101_WIZARD_STEPS];
    if (family === 'trossen') return [ROBOT_INFO_STEP, ...TROSSEN_WIZARD_STEPS];
    return [ROBOT_INFO_STEP];
}

function getUnifiedLabels(family: 'so101' | 'trossen' | null): Record<string, string> {
    if (family === 'so101') return { [ROBOT_INFO_STEP]: ROBOT_INFO_LABEL, ...SO101_STEP_LABELS };
    if (family === 'trossen') return { [ROBOT_INFO_STEP]: ROBOT_INFO_LABEL, ...TROSSEN_STEP_LABELS };
    return { [ROBOT_INFO_STEP]: ROBOT_INFO_LABEL };
}

// ---------------------------------------------------------------------------
// Submit button — adapts to robot type
// ---------------------------------------------------------------------------

/**
 * When no wizard applies (unknown type), fall back to the default POST button.
 * When SO101 or Trossen is selected, show "Begin Setup" which mounts the
 * wizard provider and starts the setup flow.
 */
const NewRobotSubmitButton = ({ onBeginSetup }: { onBeginSetup: () => void }) => {
    const robotForm = useRobotForm();
    const family = robotFamily(robotForm.type);

    if (!family) {
        return <SubmitNewRobotButton />;
    }

    const isDisabled =
        !robotForm.name ||
        !robotForm.type ||
        (family === 'so101' ? !robotForm.serial_number : !robotForm.connection_string);

    return (
        <Button variant='accent' isDisabled={isDisabled} onPress={onBeginSetup}>
            Begin Setup
        </Button>
    );
};

// ---------------------------------------------------------------------------
// Robot Info — stepper + form (no wizard provider needed)
// ---------------------------------------------------------------------------

/**
 * Left-column content for the Robot Info step. Renders the static stepper
 * and the robot form. These are Fragment children so Grid sees the
 * gridArea Views as direct items.
 */
const RobotInfoContent = ({ onBeginSetup }: { onBeginSetup: () => void }) => {
    const robotForm = useRobotForm();
    const family = robotFamily(robotForm.type);

    const unifiedSteps = getUnifiedSteps(family);
    const unifiedLabels = getUnifiedLabels(family);

    return (
        <>
            <View gridArea='stepper'>
                <Stepper
                    steps={unifiedSteps}
                    currentStep={ROBOT_INFO_STEP}
                    completedSteps={new Set<string>()}
                    labels={unifiedLabels}
                    onGoToStep={() => {
                        /* robot_info is already current; other steps aren't reachable yet */
                    }}
                />
                <Divider orientation='horizontal' size='S' marginTop='size-200' />
            </View>
            <View gridArea='form' UNSAFE_style={{ overflowY: 'auto' }} paddingBottom='size-400' minWidth={0}>
                <Suspense fallback={<CenteredLoading />}>
                    <RobotForm submitButton={<NewRobotSubmitButton onBeginSetup={onBeginSetup} />} />
                </Suspense>
            </View>
        </>
    );
};

// ---------------------------------------------------------------------------
// SO101 wizard — stepper + step body + highlight sync (inside provider)
// ---------------------------------------------------------------------------

/**
 * Left-column content for the SO101 wizard. Must be rendered inside
 * `SetupWizardProvider` because it reads wizard state via hooks.
 *
 * Also renders `SO101ViewerEffects` which drives calibration animations
 * and syncs highlights up to the parent via `onHighlightsChange`.
 */
const SO101WizardContent = ({
    onBack,
    onHighlightsChange,
}: {
    onBack: () => void;
    onHighlightsChange: (highlights: JointHighlight[]) => void;
}) => {
    const { currentStep, completedSteps } = useSetupState();
    const { visibleSteps, goToStep } = useSetupActions();

    const unifiedSteps = [ROBOT_INFO_STEP, ...visibleSteps];
    const unifiedLabels = { [ROBOT_INFO_STEP]: ROBOT_INFO_LABEL, ...SO101_STEP_LABELS };
    const unifiedCompleted = new Set<string>([ROBOT_INFO_STEP, ...completedSteps]);

    return (
        <>
            <SO101ViewerEffects onHighlightsChange={onHighlightsChange} />
            <View gridArea='stepper'>
                <Stepper
                    steps={unifiedSteps}
                    currentStep={currentStep}
                    completedSteps={unifiedCompleted}
                    labels={unifiedLabels}
                    onGoToStep={(step) => {
                        if (step === ROBOT_INFO_STEP) {
                            onBack();
                        } else {
                            goToStep(step as WizardStep);
                        }
                    }}
                />
                <Divider orientation='horizontal' size='S' marginTop='size-200' />
            </View>
            <View gridArea='form' UNSAFE_style={{ overflowY: 'auto' }} paddingBottom='size-400' minWidth={0}>
                <SO101StepBody />
            </View>
        </>
    );
};

// ---------------------------------------------------------------------------
// Trossen wizard — stepper + step body (inside provider)
// ---------------------------------------------------------------------------

/**
 * Left-column content for the Trossen wizard. Must be rendered inside
 * `TrossenSetupWizardProvider` because it reads wizard state via hooks.
 */
const TrossenWizardContent = ({ onBack }: { onBack: () => void }) => {
    const { currentStep, completedSteps } = useTrossenSetupState();
    const { visibleSteps, goToStep } = useTrossenSetupActions();

    const unifiedSteps = [ROBOT_INFO_STEP, ...visibleSteps];
    const unifiedLabels = { [ROBOT_INFO_STEP]: ROBOT_INFO_LABEL, ...TROSSEN_STEP_LABELS };
    const unifiedCompleted = new Set<string>([ROBOT_INFO_STEP, ...completedSteps]);

    return (
        <>
            <View gridArea='stepper'>
                <Stepper
                    steps={unifiedSteps}
                    currentStep={currentStep}
                    completedSteps={unifiedCompleted}
                    labels={unifiedLabels}
                    onGoToStep={(step) => {
                        if (step === ROBOT_INFO_STEP) {
                            onBack();
                        } else {
                            goToStep(step as TrossenWizardStep);
                        }
                    }}
                />
                <Divider orientation='horizontal' size='S' marginTop='size-200' />
            </View>
            <View gridArea='form' UNSAFE_style={{ overflowY: 'auto' }} paddingBottom='size-400' minWidth={0}>
                <TrossenStepBody />
            </View>
        </>
    );
};

// ---------------------------------------------------------------------------
// Unified viewer panel — always rendered, never remounts
// ---------------------------------------------------------------------------

/**
 * Right-column viewer. Rendered once in a stable tree position so the
 * expensive three.js Canvas survives robot_info <-> wizard transitions.
 *
 * Shows the dashed empty-state when no robot type is selected, otherwise
 * renders `SetupRobotViewer` with the given highlights.
 */
const UnifiedViewerPanel = ({ highlights }: { highlights: JointHighlight[] }) => {
    const robotForm = useRobotForm();
    const robotType = robotForm.type || null;

    if (!robotType) {
        return (
            <View
                height='100%'
                backgroundColor='gray-200'
                UNSAFE_style={{
                    borderRadius: 'var(--spectrum-alias-border-radius-regular)',
                    borderColor: 'var(--spectrum-global-color-gray-700)',
                    borderWidth: '1px',
                    borderStyle: 'dashed',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                }}
            >
                <Heading level={4} UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-500)' }}>
                    Select a robot type to preview
                </Heading>
            </View>
        );
    }

    return (
        <View
            height='100%'
            backgroundColor='gray-200'
            UNSAFE_style={{
                borderRadius: 'var(--spectrum-alias-border-radius-regular)',
                overflow: 'hidden',
            }}
        >
            <SetupRobotViewer robotType={robotType as SchemaRobotType} highlights={highlights} />
        </View>
    );
};

// ---------------------------------------------------------------------------
// Inner page — requires RobotFormProvider to be mounted above
// ---------------------------------------------------------------------------

/**
 * Single stable Grid shell. The viewer always occupies the same tree
 * position so the three.js Canvas is never unmounted when switching
 * between robot_info and wizard views. Only the left-column content
 * (stepper + form/step body) changes — wrapped in the appropriate
 * wizard provider when needed.
 *
 * `wizardStarted` controls whether the wizard provider is mounted.
 * It stays `false` while the user fills in the robot form, deferring
 * the websocket connection until the user clicks "Begin Setup".
 */
const NewRobotPage = () => {
    const [wizardStarted, setWizardStarted] = useState(false);
    const [highlights, setHighlights] = useState<JointHighlight[]>([]);
    const robotForm = useRobotForm();
    const family = robotFamily(robotForm.type);

    // Stable callback ref so SO101ViewerEffects doesn't re-render when
    // the parent re-renders for unrelated reasons.
    const highlightsRef = useRef(setHighlights);
    highlightsRef.current = setHighlights;
    const onHighlightsChange = useCallback((h: JointHighlight[]) => highlightsRef.current(h), []);

    const goBackToForm = useCallback(() => {
        setWizardStarted(false);
        setHighlights([]);
    }, []);

    const beginSetup = useCallback(() => setWizardStarted(true), []);

    return (
        <View height='100%' backgroundColor='gray-100' padding='size-400' UNSAFE_style={{ overflow: 'hidden' }}>
            <Grid
                areas={['stepper stepper', 'form viewer']}
                columns={['size-6000', '1fr']}
                rows={['auto', '1fr']}
                gap='size-400'
                height='100%'
                UNSAFE_className={classes.wizardGrid}
            >
                {/* Left column: stepper + form/step body */}
                {!wizardStarted && <RobotInfoContent onBeginSetup={beginSetup} />}

                {wizardStarted && family === 'so101' && (
                    <SetupWizardProvider onBackToRobotInfo={goBackToForm}>
                        <SO101WizardContent onBack={goBackToForm} onHighlightsChange={onHighlightsChange} />
                    </SetupWizardProvider>
                )}

                {wizardStarted && family === 'trossen' && (
                    <TrossenSetupWizardProvider onBackToRobotInfo={goBackToForm}>
                        <TrossenWizardContent onBack={goBackToForm} />
                    </TrossenSetupWizardProvider>
                )}

                {/* Right column: viewer — always stable, never remounts */}
                <View gridArea='viewer'>
                    <UnifiedViewerPanel highlights={highlights} />
                </View>
            </Grid>
        </View>
    );
};

// ---------------------------------------------------------------------------
// Exported route component — wraps with providers (replaces NewRobotLayout)
// ---------------------------------------------------------------------------

export const New = () => {
    return (
        <RobotModelsProvider>
            <RobotFormProvider>
                <NewRobotPage />
            </RobotFormProvider>
        </RobotModelsProvider>
    );
};
