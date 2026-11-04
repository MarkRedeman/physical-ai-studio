import { createContext, ReactNode, useContext, useRef, useState } from 'react';

import { toast } from '@geti-ui/ui';

import { SchemaTeleopControlConfig } from '../../../api/openapi-spec';
import { ERROR_TOAST, useRuntimeSession } from '../runtime-session-provider';
import { useHomePositionBusy } from './home-position-provider';
import { usePartControls } from './part-controls';

export type ModeKey = 'record' | 'inference';
type SessionCommand = Parameters<ReturnType<typeof useRuntimeSession>['send']>[0];

const useModeState = (initialMode: ModeKey) => {
    const { state, send, isLoading } = useRuntimeSession();
    const { pendingPart } = usePartControls();
    const homeBusy = useHomePositionBusy();
    const [mode, setMode] = useState(initialMode);
    const [isPending, setIsPending] = useState(false);
    const inFlight = useRef(false);
    const canSwitch = !isPending && !homeBusy && !isLoading && pendingPart === null && !state?.dataset?.is_recording;

    // Recording and model commands share a lock so mode changes cannot race them.
    const run = async (command: SessionCommand, fallback = 'The session command failed.') => {
        if (inFlight.current || homeBusy) return;
        inFlight.current = true;
        setIsPending(true);
        try {
            const ack = await send(command);
            if (!ack.ok) toast.negative(ack.error?.message ?? fallback, ERROR_TOAST);
            return ack;
        } catch (error) {
            toast.negative(error instanceof Error ? error.message : fallback, ERROR_TOAST);
        } finally {
            inFlight.current = false;
            setIsPending(false);
        }
    };

    const selectMode = async (next: ModeKey) => {
        if (next === mode || !canSwitch || inFlight.current) return;
        // Preserve passive mode selection against a backend that predates the home/idle-check API.
        const hasHomeApi = state != null && ('home_motion' in state || 'home_capability' in (state.environment ?? {}));
        if (state?.control == null && !hasHomeApi) {
            setMode(next);
            return;
        }
        const command: SessionCommand =
            state?.control != null
                ? { command: 'set_control', control: null, reject_open_episode: true }
                : { command: 'check_recording_idle' };
        const ack = await run(command, 'Could not switch session mode.');
        if (!ack?.ok) return;
        setMode(next);
    };

    const setTeleoperation = async (control: SchemaTeleopControlConfig | null) => {
        if (homeBusy || isLoading || pendingPart !== null || state?.dataset?.is_recording) return;
        const ack = await run(
            { command: 'set_control', control, reject_open_episode: true },
            'Could not switch teleoperation.'
        );
        if (ack?.ok && control != null) setMode('record');
        return ack;
    };
    // Home return has already stopped model control on the backend before this is called.
    const activateRecording = () => setMode('record');

    return { mode, selectMode, isPending: isPending || homeBusy, canSwitch, run, setTeleoperation, activateRecording };
};

const SessionModeContext = createContext<ReturnType<typeof useModeState> | null>(null);

export const SessionModeProvider = ({ initialMode, children }: { initialMode: ModeKey; children: ReactNode }) => {
    const value = useModeState(initialMode);
    return <SessionModeContext.Provider value={value}>{children}</SessionModeContext.Provider>;
};

export const useSessionMode = () => {
    const value = useContext(SessionModeContext);
    if (value === null) throw new Error('useSessionMode must be used within a SessionModeProvider');
    return value;
};
