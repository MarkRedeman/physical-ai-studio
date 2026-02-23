import { createContext, ReactNode, useCallback, useContext, useMemo, useState } from 'react';

import { useProjectId } from '../../../projects/use-project';
import { useRobotForm } from '../../robot-form/provider';
import { useTrossenDebug } from './debug-panel';
import { TrossenSetupWebSocketState, useTrossenSetupWebSocket } from './use-trossen-setup-websocket';

// ---------------------------------------------------------------------------
// Wizard step definitions — Trossen has only 2 steps
// ---------------------------------------------------------------------------

export enum TrossenWizardStep {
    /** IP reachability + driver configure check */
    DIAGNOSTICS = 'diagnostics',
    /** Live 3D preview + save robot */
    VERIFICATION = 'verification',
}

export const TROSSEN_WIZARD_STEPS: TrossenWizardStep[] = [
    TrossenWizardStep.DIAGNOSTICS,
    TrossenWizardStep.VERIFICATION,
];

export const TROSSEN_STEP_LABELS: Record<TrossenWizardStep, string> = {
    [TrossenWizardStep.DIAGNOSTICS]: 'Diagnostics',
    [TrossenWizardStep.VERIFICATION]: 'Verification',
};

// ---------------------------------------------------------------------------
// Context shapes
// ---------------------------------------------------------------------------

interface TrossenSetupState {
    currentStep: TrossenWizardStep;
    completedSteps: Set<TrossenWizardStep>;
    wsState: TrossenSetupWebSocketState;
}

interface TrossenSetupActions {
    goToStep: (step: TrossenWizardStep) => void;
    goNext: () => void;
    goBack: () => void;
    markCompleted: (step: TrossenWizardStep) => void;
    canGoNext: boolean;
    canGoBack: boolean;
    stepIndex: number;
    visibleSteps: TrossenWizardStep[];
    commands: {
        reProbe: () => void;
        enterVerification: () => void;
        ping: () => void;
    };
}

const TrossenSetupStateContext = createContext<TrossenSetupState | null>(null);
const TrossenSetupActionsContext = createContext<TrossenSetupActions | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export const TrossenSetupWizardProvider = ({ children }: { children: ReactNode }) => {
    const { project_id: projectId } = useProjectId();
    const robotForm = useRobotForm();

    // -----------------------------------------------------------------------
    // Wizard step state
    // -----------------------------------------------------------------------

    const [wizardState, setWizardState] = useState({
        currentStep: TrossenWizardStep.DIAGNOSTICS as TrossenWizardStep,
        completedSteps: new Set<TrossenWizardStep>(),
    });

    const visibleSteps = TROSSEN_WIZARD_STEPS;
    const stepIndex = visibleSteps.indexOf(wizardState.currentStep);

    const goToStep = useCallback((step: TrossenWizardStep) => {
        setWizardState((prev) => ({ ...prev, currentStep: step }));
    }, []);

    const goNext = useCallback(() => {
        setWizardState((prev) => {
            const idx = TROSSEN_WIZARD_STEPS.indexOf(prev.currentStep);
            if (idx < TROSSEN_WIZARD_STEPS.length - 1) {
                return { ...prev, currentStep: TROSSEN_WIZARD_STEPS[idx + 1] };
            }
            return prev;
        });
    }, []);

    const goBack = useCallback(() => {
        setWizardState((prev) => {
            const idx = TROSSEN_WIZARD_STEPS.indexOf(prev.currentStep);
            if (idx > 0) {
                return { ...prev, currentStep: TROSSEN_WIZARD_STEPS[idx - 1] };
            }
            return prev;
        });
    }, []);

    const markCompleted = useCallback((step: TrossenWizardStep) => {
        setWizardState((prev) => {
            const next = new Set(prev.completedSteps);
            next.add(step);
            return { ...prev, completedSteps: next };
        });
    }, []);

    // -----------------------------------------------------------------------
    // WebSocket hook — uses mock state from TrossenDebugProvider when active
    // -----------------------------------------------------------------------

    const debug = useTrossenDebug();
    const isDebug = debug?.isDebug ?? false;

    const connectionString = robotForm.connection_string ?? '';
    const robotType = robotForm.type ?? '';
    const wsEnabled = !isDebug && !!connectionString && !!robotType;

    const realWs = useTrossenSetupWebSocket({
        projectId,
        robotType,
        connectionString,
        enabled: wsEnabled,
    });

    const wsState = isDebug && debug ? debug.mockState : realWs.state;
    const commands = isDebug && debug ? debug.commands : realWs.commands;

    // -----------------------------------------------------------------------
    // Context values
    // -----------------------------------------------------------------------

    const state: TrossenSetupState = {
        currentStep: wizardState.currentStep,
        completedSteps: wizardState.completedSteps,
        wsState,
    };

    const actions: TrossenSetupActions = useMemo(
        () => ({
            goToStep,
            goNext,
            goBack,
            markCompleted,
            canGoNext: stepIndex < visibleSteps.length - 1,
            canGoBack: stepIndex > 0,
            stepIndex,
            visibleSteps,
            commands,
        }),
        [goToStep, goNext, goBack, markCompleted, stepIndex, visibleSteps, commands]
    );

    return (
        <TrossenSetupStateContext.Provider value={state}>
            <TrossenSetupActionsContext.Provider value={actions}>{children}</TrossenSetupActionsContext.Provider>
        </TrossenSetupStateContext.Provider>
    );
};

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

export const useTrossenSetupState = () => {
    const ctx = useContext(TrossenSetupStateContext);
    if (ctx === null) throw new Error('useTrossenSetupState must be used within TrossenSetupWizardProvider');
    return ctx;
};

export const useTrossenSetupActions = () => {
    const ctx = useContext(TrossenSetupActionsContext);
    if (ctx === null) throw new Error('useTrossenSetupActions must be used within TrossenSetupWizardProvider');
    return ctx;
};
