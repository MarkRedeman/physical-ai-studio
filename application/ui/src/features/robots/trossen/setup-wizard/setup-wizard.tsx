import { Divider, Grid, Heading, View } from '@geti/ui';

import { SchemaRobotType } from '../../../../api/openapi-spec';
import { useRobotForm } from '../../robot-form/provider';
import { SetupRobotViewer } from '../../shared/setup-wizard/setup-robot-viewer';
import { Stepper } from '../../shared/setup-wizard/stepper';
import { TrossenDiagnosticsStep } from './diagnostics-step';
import { TrossenVerificationStep } from './verification-step';
import {
    TROSSEN_STEP_LABELS,
    TrossenWizardStep,
    useTrossenSetupActions,
    useTrossenSetupState,
} from './wizard-provider';

import classes from '../../shared/setup-wizard/setup-wizard.module.scss';

// ---------------------------------------------------------------------------
// Right-column viewer panel
// ---------------------------------------------------------------------------

/**
 * Right column: shows the 3D URDF viewer.
 * No animations for Trossen — just the static/live-synced model.
 */
const ViewerPanel = () => {
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
            <SetupRobotViewer robotType={robotType as SchemaRobotType} highlights={[]} />
        </View>
    );
};

// ---------------------------------------------------------------------------
// Main wizard content
// ---------------------------------------------------------------------------

/**
 * Trossen setup wizard — two-column layout (same structure as SO101).
 * Left: stepper + current step content.
 * Right: 3D robot viewer (no highlights or animations for Trossen).
 *
 * Only 2 steps: DIAGNOSTICS → VERIFICATION.
 */
export const TrossenSetupWizardContent = () => {
    const { currentStep, completedSteps } = useTrossenSetupState();
    const { visibleSteps, goToStep } = useTrossenSetupActions();

    return (
        <Grid
            areas={['stepper stepper', 'form viewer']}
            columns={['size-6000', '1fr']}
            rows={['auto', '1fr']}
            gap='size-400'
            height='100%'
            UNSAFE_className={classes.wizardGrid}
        >
            {/* Top row: stepper spans full width */}
            <View gridArea='stepper'>
                <Stepper
                    steps={visibleSteps}
                    currentStep={currentStep}
                    completedSteps={completedSteps}
                    labels={TROSSEN_STEP_LABELS}
                    onGoToStep={goToStep}
                />
                <Divider orientation='horizontal' size='S' marginTop='size-200' />
            </View>

            {/* Left column: current step content */}
            <View gridArea='form' UNSAFE_style={{ overflowY: 'auto' }} paddingBottom='size-400' minWidth={0}>
                {currentStep === TrossenWizardStep.DIAGNOSTICS && <TrossenDiagnosticsStep />}
                {currentStep === TrossenWizardStep.VERIFICATION && <TrossenVerificationStep />}
            </View>

            {/* Right column: 3D robot viewer */}
            <View gridArea='viewer'>
                <ViewerPanel />
            </View>
        </Grid>
    );
};
