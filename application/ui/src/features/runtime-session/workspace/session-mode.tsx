import { createContext, ReactNode, useContext, useRef, useState } from 'react';

import { toast } from '@geti-ui/ui';

import { ERROR_TOAST, useRuntimeSession } from '../runtime-session-provider';
import { usePartControls } from './part-controls';

export type ModeKey = 'record' | 'inference';
type SessionCommand = Parameters<ReturnType<typeof useRuntimeSession>['send']>[0];

const useModeState = (initialMode: ModeKey) => {
    const { state, send, isLoading } = useRuntimeSession();
    const { pendingPart } = usePartControls();
    const [mode, setMode] = useState(initialMode);
    const [isPending, setIsPending] = useState(false);
    const inFlight = useRef(false);
    const canSwitch = !isPending && !isLoading && pendingPart === null && !state?.dataset?.is_recording;

    // Recording and model commands share a lock so mode changes cannot race them.
    const run = async (command: SessionCommand, fallback = 'The session command failed.') => {
        if (inFlight.current) return;
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
        if (state?.control != null) {
            const ack = await run({ command: 'set_control', control: null }, 'Could not switch session mode.');
            if (!ack?.ok) return;
        }
        setMode(next);
    };

    return { mode, selectMode, isPending, canSwitch, run };
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
