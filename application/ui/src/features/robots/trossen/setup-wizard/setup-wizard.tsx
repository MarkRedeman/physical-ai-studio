import { Heading, View } from '@geti/ui';

import { SchemaRobotType } from '../../../../api/openapi-spec';
import { useRobotForm } from '../../robot-form/provider';
import { SetupRobotViewer } from '../../shared/setup-wizard/setup-robot-viewer';
import { TrossenDiagnosticsStep } from './diagnostics-step';
import { TrossenVerificationStep } from './verification-step';
import { TrossenWizardStep, useTrossenSetupState } from './wizard-provider';

// ---------------------------------------------------------------------------
// Right-column viewer panel
// ---------------------------------------------------------------------------

/**
 * Right column: shows the 3D URDF viewer.
 * No animations for Trossen — just the static/live-synced model.
 */
export const TrossenViewerPanel = () => {
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
// Step body — renders the current step's form content (no stepper, no viewer)
// ---------------------------------------------------------------------------

/**
 * Renders the form/content for the current Trossen wizard step.
 * Used by the unified /robots/new page which owns the stepper and layout.
 */
export const TrossenStepBody = () => {
    const { currentStep } = useTrossenSetupState();

    return (
        <>
            {currentStep === TrossenWizardStep.DIAGNOSTICS && <TrossenDiagnosticsStep />}
            {currentStep === TrossenWizardStep.VERIFICATION && <TrossenVerificationStep />}
        </>
    );
};
