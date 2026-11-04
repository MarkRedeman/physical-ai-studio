import { createContext, ReactNode, useContext, useEffect, useRef, useState } from 'react';

import { toast } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import { SchemaHomeMotion, SchemaHomePose } from '../../../api/openapi-spec';
import { useProjectId } from '../../projects/use-project';
import { ERROR_TOAST, failureOf, useRuntimeSession } from '../runtime-session-provider';
import { decodeStoredHome, homeStorageKey, validateHomePose } from './home-position-storage';

interface StoredHome {
    identity: string;
    pose: SchemaHomePose | null;
    invalid: boolean;
    warning: string | null;
}

const parseMotion = (value: unknown): SchemaHomeMotion | null => {
    if (value == null || typeof value !== 'object') return null;
    const motion = value as Record<string, unknown>;
    if (
        typeof motion.operation_id !== 'string' ||
        typeof motion.robot_id !== 'string' ||
        !['moving', 'reached', 'cancelled', 'failed'].includes(String(motion.state)) ||
        typeof motion.progress !== 'number' ||
        !Number.isFinite(motion.progress)
    )
        return null;
    return {
        operation_id: motion.operation_id,
        robot_id: motion.robot_id,
        state: motion.state as SchemaHomeMotion['state'],
        progress: motion.progress,
        sequence: typeof motion.sequence === 'number' ? motion.sequence : 0,
        message: typeof motion.message === 'string' ? motion.message : null,
    };
};

const useHomeState = () => {
    const { project_id } = useProjectId();
    const { state, statuses, send, isLoading } = useRuntimeSession();
    const environment = state?.environment;
    const capability = environment?.home_capability ?? null;
    const key = environment ? homeStorageKey(project_id, environment) : null;
    const identity = [state?.session_id ?? '', key ?? environment?.robot_id ?? '', capability?.fingerprint ?? ''].join(
        ':'
    );
    const [stored, setStored] = useState<StoredHome>({ identity: '', pose: null, invalid: false, warning: null });
    const [phase, setPhase] = useState<'idle' | 'capturing' | 'starting' | 'stopping'>('idle');
    const [error, setError] = useState<string | null>(null);
    const [localMotion, setLocalMotion] = useState<SchemaHomeMotion | null>(null);
    const [requestPending, setRequestPending] = useState(false);
    const [dismissedOutcome, setDismissedOutcome] = useState<string | null>(null);
    const operation = useRef<string | null>(null);
    const renewing = useRef(false);
    const leaseEnabled = useRef(false);
    const activeOperation = useRef<string | null>(null);
    const sessionPresent = useRef(false);
    const requestInFlight = useRef(false);
    const cancellationInFlight = useRef(false);

    useEffect(() => {
        setLocalMotion(null);
        setError(null);
        operation.current = null;
        leaseEnabled.current = false;
    }, [identity]);

    useEffect(() => {
        setStored((previous) => {
            if (previous.identity === identity) return previous;
            if (capability == null || key == null) return { identity, pose: null, invalid: false, warning: null };
            try {
                const serialized = window.localStorage.getItem(key);
                const pose = decodeStoredHome(serialized, capability);
                return { identity, pose, invalid: serialized != null && pose == null, warning: null };
            } catch {
                return {
                    identity,
                    pose: null,
                    invalid: false,
                    warning: 'Browser storage is unavailable. Home can be used for this session only.',
                };
            }
        });
    }, [identity, key, capability]);

    const current = stored.identity === identity ? stored : { identity, pose: null, invalid: false, warning: null };
    const serverMotion =
        state?.home_motion?.robot_id === environment?.robot_id &&
        (phase !== 'starting' || state?.home_motion?.operation_id === operation.current)
            ? (state?.home_motion ?? null)
            : null;
    let motion = localMotion?.robot_id === environment?.robot_id ? localMotion : null;
    if (
        serverMotion &&
        (motion == null ||
            (serverMotion.operation_id === motion.operation_id &&
                (serverMotion.sequence ?? 0) >= (motion.sequence ?? 0)))
    )
        motion = serverMotion;
    const isReturning = phase === 'starting' || phase === 'stopping' || motion?.state === 'moving';
    const isBusy = requestPending || phase !== 'idle' || isReturning;
    const isRecording = state?.dataset?.is_recording ?? false;
    const isReady =
        capability != null &&
        !isLoading &&
        state != null &&
        statuses['environment/follower']?.state === 'running' &&
        failureOf(statuses, 'environment') === null;
    const canCapture = isReady && !isBusy && !isRecording && state?.control == null;
    const canReturn = isReady && !isBusy && !isRecording && current.pose != null;

    const report = (message: string) => {
        setError(message);
        toast.negative(message, ERROR_TOAST);
    };
    const acceptMotion = (next: SchemaHomeMotion) =>
        setLocalMotion((previous) =>
            previous?.operation_id === next.operation_id && (previous.sequence ?? 0) > (next.sequence ?? 0)
                ? previous
                : next
        );

    const capture = async () => {
        if (!canCapture || capability == null || environment == null || requestInFlight.current) return;
        requestInFlight.current = true;
        setRequestPending(true);
        setPhase('capturing');
        setError(null);
        try {
            const ack = await send({ command: 'capture_home_position', robot_id: environment.robot_id });
            if (!ack.ok) {
                report(ack.error?.message ?? 'Could not capture home.');
                return;
            }
            const pose = validateHomePose(ack.result, capability);
            if (pose == null) {
                report('The captured home pose did not pass validation.');
                return;
            }
            let warning: string | null = key == null ? 'Home is available for this session only.' : null;
            if (key != null) {
                try {
                    window.localStorage.setItem(key, JSON.stringify(pose));
                } catch {
                    warning = 'Could not save in this browser. Home is available for this session only.';
                }
            }
            setStored({ identity, pose, invalid: false, warning });
            setDismissedOutcome(motion?.operation_id ?? null);
        } catch (reason) {
            report(reason instanceof Error ? reason.message : 'Could not capture home.');
        } finally {
            requestInFlight.current = false;
            setRequestPending(false);
            setPhase((previous) => (previous === 'stopping' ? previous : 'idle'));
        }
    };

    const start = async (): Promise<boolean> => {
        if (!canReturn || current.pose == null || requestInFlight.current) return false;
        requestInFlight.current = true;
        setRequestPending(true);
        const operation_id = uuidv4();
        operation.current = operation_id;
        leaseEnabled.current = true;
        setLocalMotion(null);
        setDismissedOutcome(null);
        setPhase('starting');
        setError(null);
        try {
            const ack = await send({ command: 'return_home', operation_id, home: current.pose, path_confirmed: true });
            if (!ack.ok) {
                leaseEnabled.current = false;
                report(ack.error?.message ?? 'Could not return home.');
                return false;
            }
            const result = parseMotion(ack.result);
            if (result) acceptMotion(result);
            return result == null || result.state === 'moving' || result.state === 'reached';
        } catch (reason) {
            leaseEnabled.current = false;
            report(reason instanceof Error ? reason.message : 'Could not start the home return.');
            return false;
        } finally {
            requestInFlight.current = false;
            setRequestPending(false);
            setPhase((previous) => (previous === 'stopping' ? previous : 'idle'));
        }
    };

    const cancel = async () => {
        const operation_id = phase === 'starting' ? operation.current : (motion?.operation_id ?? operation.current);
        if (operation_id == null || cancellationInFlight.current) return;
        cancellationInFlight.current = true;
        leaseEnabled.current = false; // Watchdog stops motion even if the cancellation message cannot arrive.
        setPhase('stopping');
        try {
            const ack = await send({ command: 'cancel_home_return', operation_id });
            if (!ack.ok) {
                report(
                    ack.error?.message ??
                        'Stop could not be confirmed. Check the robot and use its emergency stop if needed.'
                );
                return;
            }
            const result = parseMotion(ack.result);
            if (result) acceptMotion(result);
        } catch (reason) {
            report(
                reason instanceof Error
                    ? reason.message
                    : 'Stop could not be confirmed. Use the physical emergency stop if needed.'
            );
        } finally {
            cancellationInFlight.current = false;
            setPhase('idle');
        }
    };

    activeOperation.current = isReturning
        ? phase === 'starting'
            ? operation.current
            : (motion?.operation_id ?? operation.current)
        : null;
    sessionPresent.current = state != null;
    useEffect(() => {
        if (!isReturning) return;
        const renew = () => {
            const operation_id = activeOperation.current;
            if (
                !leaseEnabled.current ||
                renewing.current ||
                operation_id == null ||
                document.visibilityState === 'hidden'
            )
                return;
            renewing.current = true;
            void send({ command: 'renew_home_return', operation_id })
                .catch(() => {
                    leaseEnabled.current = false;
                })
                .finally(() => {
                    renewing.current = false;
                });
        };
        renew();
        const interval = window.setInterval(renew, 250);
        const hidden = () => {
            if (document.visibilityState === 'hidden') {
                leaseEnabled.current = false;
                const operation_id = activeOperation.current;
                if (operation_id) void send({ command: 'cancel_home_return', operation_id }).catch(() => {});
            }
        };
        document.addEventListener('visibilitychange', hidden);
        return () => {
            window.clearInterval(interval);
            document.removeEventListener('visibilitychange', hidden);
        };
    }, [isReturning, send]);
    useEffect(
        () => () => {
            leaseEnabled.current = false;
            if (activeOperation.current && sessionPresent.current)
                void send({ command: 'cancel_home_return', operation_id: activeOperation.current }).catch(() => {});
        },
        [send]
    );

    const clear = () => {
        if (isBusy || isRecording) return;
        let warning: string | null = null;
        if (key != null) {
            try {
                window.localStorage.removeItem(key);
            } catch {
                warning = 'Could not remove the saved pose from browser storage.';
            }
        }
        setStored({ identity, pose: null, invalid: false, warning });
        setDismissedOutcome(motion?.operation_id ?? null);
        setError(null);
    };

    return {
        pose: current.pose,
        invalid: current.invalid,
        warning: current.warning,
        capability,
        phase,
        error,
        motion,
        feedback: motion?.operation_id === dismissedOutcome ? null : motion,
        isReturning,
        isBusy,
        canCapture,
        canReturn,
        isRecording,
        capture,
        start,
        cancel,
        clear,
    };
};

const HomePositionContext = createContext<ReturnType<typeof useHomeState> | null>(null);

export const HomePositionProvider = ({ children }: { children: ReactNode }) => {
    const value = useHomeState();
    return <HomePositionContext.Provider value={value}>{children}</HomePositionContext.Provider>;
};

export const useHomePosition = () => {
    const value = useContext(HomePositionContext);
    if (value == null) throw new Error('useHomePosition must be used within a HomePositionProvider');
    return value;
};

export const useHomePositionBusy = () => useContext(HomePositionContext)?.isBusy ?? false;
