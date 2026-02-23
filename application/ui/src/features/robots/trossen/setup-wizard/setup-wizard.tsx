import { TrossenDiagnosticsStep } from './diagnostics-step';
import { TrossenVerificationStep } from './verification-step';
import { TrossenWizardStep, useTrossenSetupState } from './wizard-provider';

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
