import { useEffect, useMemo, useRef } from 'react';

import { JointHighlight } from '../shared/use-joint-highlight';
import { CalibrationStep } from './calibration-step';
import { DiagnosticsStep } from './diagnostics-step';
import { MotorSetupStep } from './motor-setup-step';
import { useCenteringAnimation, useRangeOfMotionAnimation } from './use-calibration-animations';
import { VerificationStep } from './verification-step';
import { useSetupState, WizardStep } from './wizard-provider';

// ---------------------------------------------------------------------------
// Motor setup order (gripper first) — matches lerobot's setup flow
// ---------------------------------------------------------------------------

const MOTOR_SETUP_ORDER = ['gripper', 'wrist_roll', 'wrist_flex', 'elbow_flex', 'shoulder_lift', 'shoulder_pan'];

// ---------------------------------------------------------------------------
// Derive highlights from setup state (replaces the old useState + useEffect)
// ---------------------------------------------------------------------------

function useHighlights(): JointHighlight[] {
    const { currentStep, wsState, preVerifyProbeResult } = useSetupState();

    return useMemo(() => {
        if (currentStep !== WizardStep.MOTOR_SETUP) {
            return [];
        }

        const progress = wsState.motorSetupProgress;

        // Derive current motor index (same logic as motor-setup-step)
        const currentMotorIndex = (() => {
            const idx = MOTOR_SETUP_ORDER.findIndex((motor) => progress[motor]?.status !== 'success');
            return idx === -1 ? MOTOR_SETUP_ORDER.length : idx;
        })();

        const allDone = currentMotorIndex >= MOTOR_SETUP_ORDER.length;

        if (allDone) {
            // Derive reassembly state
            const hasTriggeredVerify = preVerifyProbeResult !== null;
            const newResultArrived = hasTriggeredVerify && wsState.probeResult !== preVerifyProbeResult;

            if (newResultArrived) {
                // After verification: color each motor based on probe result
                const probeMotors = wsState.probeResult?.motors ?? [];
                return probeMotors.map((m) => ({
                    joint: m.name,
                    color: m.found ? ('positive' as const) : ('negative' as const),
                }));
            }
            // Reassembly idle/verifying: highlight all motors in accent
            return MOTOR_SETUP_ORDER.map((j) => ({ joint: j, color: 'accent' as const }));
        }

        const currentMotor = MOTOR_SETUP_ORDER[currentMotorIndex];
        if (currentMotor) {
            return [{ joint: currentMotor, color: 'accent' as const }];
        }

        return [];
    }, [currentStep, wsState.motorSetupProgress, wsState.probeResult, preVerifyProbeResult]);
}

// ---------------------------------------------------------------------------
// Viewer effects — drives animations and syncs highlights to the parent
// ---------------------------------------------------------------------------

/**
 * Renderless component that must live inside `SetupWizardProvider`.
 * It derives viewer highlights from wizard state, drives calibration
 * animations, and pushes the current highlights up to the parent
 * (which owns the viewer) via the `onHighlightsChange` callback.
 *
 * This allows the viewer to live outside the wizard provider (preventing
 * expensive Canvas remounts on page view transitions) while still
 * receiving highlight data from inside the provider.
 */
export const SO101ViewerEffects = ({
    onHighlightsChange,
}: {
    onHighlightsChange: (highlights: JointHighlight[]) => void;
}) => {
    const { currentStep, calibrationPhase } = useSetupState();
    const highlights = useHighlights();

    const isCentering =
        currentStep === WizardStep.CALIBRATION &&
        (calibrationPhase === 'instructions' || calibrationPhase === 'homing');

    const isRangeDemo = currentStep === WizardStep.CALIBRATION && calibrationPhase === 'recording';

    // Drive animations — these hooks are no-ops when `enabled` is false
    useCenteringAnimation(isCentering);
    useRangeOfMotionAnimation(isRangeDemo);

    // Sync highlights to the parent. Only fire when the derived highlights
    // actually change (keyed on the serialized value to avoid spurious calls
    // from new-but-equal array references).
    const highlightsKey = highlights.map((h) => `${h.joint}:${h.color}`).join(',');
    const prevKeyRef = useRef(highlightsKey);
    const callbackRef = useRef(onHighlightsChange);
    callbackRef.current = onHighlightsChange;

    // Sync on mount and whenever the derived highlights change
    useEffect(() => {
        callbackRef.current(highlights);
        prevKeyRef.current = highlightsKey;
    }, [highlightsKey, highlights]);

    // Clear highlights on unmount (wizard provider being torn down)
    useEffect(() => {
        return () => {
            callbackRef.current([]);
        };
    }, []);

    return null;
};

// ---------------------------------------------------------------------------
// Step body — renders the current step's form content (no stepper, no viewer)
// ---------------------------------------------------------------------------

/**
 * Renders the form/content for the current SO101 wizard step.
 * Used by the unified /robots/new page which owns the stepper and layout.
 */
export const SO101StepBody = () => {
    const { currentStep } = useSetupState();

    return (
        <>
            {currentStep === WizardStep.DIAGNOSTICS && <DiagnosticsStep />}
            {currentStep === WizardStep.MOTOR_SETUP && <MotorSetupStep />}
            {currentStep === WizardStep.CALIBRATION && <CalibrationStep />}
            {currentStep === WizardStep.VERIFICATION && <VerificationStep />}
        </>
    );
};
