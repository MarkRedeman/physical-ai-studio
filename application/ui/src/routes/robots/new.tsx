import { Suspense, useState } from 'react';

import { Button, Divider, Flex, Grid, Loading, View } from '@geti/ui';

import { RobotForm } from '../../features/robots/robot-form/form';
import { Preview } from '../../features/robots/robot-form/preview';
import { RobotFormProvider, useRobotForm } from '../../features/robots/robot-form/provider';
import { SubmitNewRobotButton } from '../../features/robots/robot-form/submit-new-robot-button';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { Stepper } from '../../features/robots/shared/setup-wizard/stepper';
import { SO101StepBody, SO101ViewerPanel } from '../../features/robots/so101/setup-wizard/setup-wizard';
import {
    SetupWizardProvider,
    STEP_LABELS as SO101_STEP_LABELS,
    WIZARD_STEPS as SO101_WIZARD_STEPS,
    useSetupActions,
    useSetupState,
    WizardStep,
} from '../../features/robots/so101/setup-wizard/wizard-provider';
import { TrossenDebugProvider } from '../../features/robots/trossen/setup-wizard/debug-panel';
import { TrossenStepBody, TrossenViewerPanel } from '../../features/robots/trossen/setup-wizard/setup-wizard';
import {
    TROSSEN_STEP_LABELS,
    TROSSEN_WIZARD_STEPS,
    TrossenSetupWizardProvider,
    TrossenWizardStep,
    useTrossenSetupActions,
    useTrossenSetupState,
} from '../../features/robots/trossen/setup-wizard/wizard-provider';

import classes from '../../features/robots/shared/setup-wizard/setup-wizard.module.scss';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * The unified page always starts on 'robot_info'. Once the user clicks
 * "Begin Setup", we switch to 'wizard' which mounts the appropriate
 * wizard provider and delegates step management to it.
 */
type PageView = 'robot_info' | 'wizard';

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
 * When SO101 or Trossen is selected, show "Begin Setup" which advances
 * the page view from 'robot_info' into the wizard.
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
// SO101 wizard wrapper — renders inside SetupWizardProvider
// ---------------------------------------------------------------------------

const SO101WizardView = ({ onBack }: { onBack: () => void }) => {
    return (
        <SetupWizardProvider>
            <SO101WizardInner onBack={onBack} />
        </SetupWizardProvider>
    );
};

/**
 * Must be a separate component so it can call SO101 wizard hooks
 * (which require SetupWizardProvider to be mounted above it).
 */
const SO101WizardInner = ({ onBack }: { onBack: () => void }) => {
    const { currentStep, completedSteps } = useSetupState();
    const { visibleSteps, goToStep } = useSetupActions();

    const unifiedSteps = [ROBOT_INFO_STEP, ...visibleSteps];
    const unifiedLabels = { [ROBOT_INFO_STEP]: ROBOT_INFO_LABEL, ...SO101_STEP_LABELS };
    const unifiedCompleted = new Set<string>([ROBOT_INFO_STEP, ...completedSteps]);

    return (
        <Grid
            areas={['stepper stepper', 'form viewer']}
            columns={['size-6000', '1fr']}
            rows={['auto', '1fr']}
            gap='size-400'
            height='100%'
            UNSAFE_className={classes.wizardGrid}
        >
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
            <View gridArea='viewer'>
                <SO101ViewerPanel />
            </View>
        </Grid>
    );
};

// ---------------------------------------------------------------------------
// Trossen wizard wrapper — renders inside TrossenSetupWizardProvider
// ---------------------------------------------------------------------------

const TrossenWizardView = ({ onBack }: { onBack: () => void }) => {
    return (
        <TrossenSetupWizardProvider>
            <TrossenWizardInner onBack={onBack} />
        </TrossenSetupWizardProvider>
    );
};

const TrossenWizardInner = ({ onBack }: { onBack: () => void }) => {
    const { currentStep, completedSteps } = useTrossenSetupState();
    const { visibleSteps, goToStep } = useTrossenSetupActions();

    const unifiedSteps = [ROBOT_INFO_STEP, ...visibleSteps];
    const unifiedLabels = { [ROBOT_INFO_STEP]: ROBOT_INFO_LABEL, ...TROSSEN_STEP_LABELS };
    const unifiedCompleted = new Set<string>([ROBOT_INFO_STEP, ...completedSteps]);

    return (
        <Grid
            areas={['stepper stepper', 'form viewer']}
            columns={['size-6000', '1fr']}
            rows={['auto', '1fr']}
            gap='size-400'
            height='100%'
            UNSAFE_className={classes.wizardGrid}
        >
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
            <View gridArea='viewer'>
                <TrossenViewerPanel />
            </View>
        </Grid>
    );
};

// ---------------------------------------------------------------------------
// Inner page — requires RobotFormProvider to be mounted above
// ---------------------------------------------------------------------------

const NewRobotPage = () => {
    const [pageView, setPageView] = useState<PageView>('robot_info');
    const robotForm = useRobotForm();
    const family = robotFamily(robotForm.type);

    const goBackToForm = () => setPageView('robot_info');
    const beginSetup = () => setPageView('wizard');

    // -----------------------------------------------------------------------
    // Wizard view — conditionally mount the appropriate provider
    // -----------------------------------------------------------------------
    if (pageView === 'wizard' && family === 'so101') {
        return (
            <View height='100%' backgroundColor='gray-100' padding='size-400' UNSAFE_style={{ overflow: 'hidden' }}>
                <SO101WizardView onBack={goBackToForm} />
            </View>
        );
    }

    if (pageView === 'wizard' && family === 'trossen') {
        return (
            <View height='100%' backgroundColor='gray-100' padding='size-400' UNSAFE_style={{ overflow: 'hidden' }}>
                <TrossenWizardView onBack={goBackToForm} />
            </View>
        );
    }

    // -----------------------------------------------------------------------
    // Robot Info view — the form + preview with stepper above
    // -----------------------------------------------------------------------
    const unifiedSteps = getUnifiedSteps(family);
    const unifiedLabels = getUnifiedLabels(family);

    return (
        <Grid
            areas={['stepper stepper', 'form viewer']}
            columns={['size-6000', '1fr']}
            rows={['auto', '1fr']}
            gap='size-400'
            height='100%'
            UNSAFE_className={classes.wizardGrid}
        >
            <View gridArea='stepper' paddingX='size-400' paddingTop='size-400'>
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
            <View gridArea='form' backgroundColor='gray-100' padding='size-400'>
                <Suspense fallback={<CenteredLoading />}>
                    <RobotForm submitButton={<NewRobotSubmitButton onBeginSetup={beginSetup} />} />
                </Suspense>
            </View>
            <View gridArea='viewer' backgroundColor='gray-50' padding='size-400'>
                <Preview />
            </View>
        </Grid>
    );
};

// ---------------------------------------------------------------------------
// Exported route component — wraps with providers (replaces NewRobotLayout)
// ---------------------------------------------------------------------------

export const New = () => {
    return (
        <RobotModelsProvider>
            <RobotFormProvider>
                <TrossenDebugProvider>
                    <NewRobotPage />
                </TrossenDebugProvider>
            </RobotFormProvider>
        </RobotModelsProvider>
    );
};
