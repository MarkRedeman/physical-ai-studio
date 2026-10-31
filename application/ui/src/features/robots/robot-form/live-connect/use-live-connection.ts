import { useCallback, useEffect, useRef, useState } from 'react';

import useWebSocket, { ReadyState } from 'react-use-websocket';
import { v4 as uuidv4 } from 'uuid';

import { fetchClient } from '../../../../api/client';
import { RobotPayload } from '../provider';

export type ConnectionPhase = 'off' | 'connecting' | 'connected' | 'releasing' | 'error';
interface ConnectionState {
    phase: ConnectionPhase;
    key: string | null;
    joints: Record<string, number>;
    error: string | null;
}
interface ConnectionRequest {
    key: string;
    robot: { id: string; name: string; type: string; payload: RobotPayload };
}
interface ReleaseWaiter {
    id: string;
    resolve: () => void;
    reject: (error: Error) => void;
}
const EMPTY_STATE: ConnectionState = { phase: 'off', key: null, joints: {}, error: null };
const CONNECTION_TIMEOUT_MS = 20_000;
const RELEASE_TIMEOUT_MS = 15_000;

export const useLiveConnection = (projectId: string, request: ConnectionRequest | null) => {
    const [state, setState] = useState(EMPTY_STATE);
    const [socketAttempt, setSocketAttempt] = useState(0);
    const [clientId] = useState(() => uuidv4());
    const active = useRef<{ id: string; key: string } | null>(null);
    const releaseWaiter = useRef<ReleaseWaiter | null>(null);
    const releasing = useRef<Promise<void> | null>(null);
    const queue = useRef(Promise.resolve());
    const mounted = useRef(true);
    const generation = useRef(0);
    const paused = useRef(false);
    const blocked = useRef(false);
    const currentRequest = useRef(request);

    const fail = useCallback((message: string) => {
        if (mounted.current) setState((previous) => ({ ...previous, phase: 'error', joints: {}, error: message }));
    }, []);

    const url = `${fetchClient.PATH('/api/projects/{project_id}/robots/draft/observations/ws', {
        params: { path: { project_id: projectId } },
    })}?client=${clientId}&attempt=${socketAttempt}`;

    const { sendJsonMessage, readyState } = useWebSocket(url, {
        share: false,
        shouldReconnect: () => false,
        onMessage: (event) => {
            try {
                const message = JSON.parse(event.data) as Record<string, unknown>;
                if (message.event === 'disconnected' && message.request_id === releaseWaiter.current?.id) {
                    releaseWaiter.current?.resolve();
                    return;
                }
                if (message.event === 'error' && message.request_id === releaseWaiter.current?.id) {
                    releaseWaiter.current?.reject(
                        new Error(typeof message.message === 'string' ? message.message : 'Robot disconnect failed.')
                    );
                    return;
                }
                const current = active.current;
                if (current === null || message.request_id !== current.id) return;
                if (message.event === 'error') {
                    if (message.error_code === 'robot_disconnect_failed') blocked.current = true;
                    fail(typeof message.message === 'string' ? message.message : 'Could not connect to the robot.');
                } else if (
                    message.event === 'observation' &&
                    current.key === currentRequest.current?.key &&
                    typeof message.data === 'object' &&
                    message.data !== null &&
                    !Array.isArray(message.data)
                ) {
                    const entries = Object.entries(message.data);
                    if (
                        entries.length === 0 ||
                        entries.some(([, value]) => typeof value !== 'number' || !Number.isFinite(value))
                    ) {
                        fail('The robot returned invalid joint readings.');
                        return;
                    }
                    setState({
                        phase: 'connected',
                        key: current.key,
                        joints: Object.fromEntries(entries),
                        error: null,
                    });
                }
            } catch {
                fail('The robot connection returned an invalid message.');
            }
        },
        onClose: () => {
            const error = new Error('Connection closed. Hardware release could not be confirmed.');
            releaseWaiter.current?.reject(error);
            if (active.current !== null) {
                blocked.current = true;
                if (mounted.current)
                    setState((previous) => ({
                        ...previous,
                        phase: 'error',
                        joints: {},
                        error: previous.error ?? error.message,
                    }));
            }
        },
        onError: () => fail('Could not reach Studio. Check the connection and retry.'),
    });

    const release = useCallback((): Promise<void> => {
        if (releasing.current !== null) return releasing.current;
        if (active.current === null) return Promise.resolve();
        if (readyState !== ReadyState.OPEN)
            return Promise.reject(new Error('Hardware release could not be confirmed.'));
        active.current = null;
        if (mounted.current) setState({ ...EMPTY_STATE, phase: 'releasing' });
        const id = uuidv4();
        const promise = new Promise<void>((resolve, reject) => {
            releaseWaiter.current = { id, resolve, reject };
            sendJsonMessage({ command: 'disconnect', request_id: id }, false);
        });
        const timer = window.setTimeout(
            () =>
                releaseWaiter.current?.reject(
                    new Error('Robot disconnect timed out. Restart Studio before reconnecting.')
                ),
            RELEASE_TIMEOUT_MS
        );
        releasing.current = promise
            .finally(() => {
                window.clearTimeout(timer);
                releaseWaiter.current = null;
                releasing.current = null;
            })
            .catch((error: unknown) => {
                blocked.current = true;
                throw error;
            });
        return releasing.current;
    }, [readyState, sendJsonMessage]);

    const enqueue = useCallback((operation: () => Promise<void>) => {
        const next = queue.current.catch(() => undefined).then(operation);
        queue.current = next;
        return next;
    }, []);

    useEffect(() => {
        currentRequest.current = request;
        const version = ++generation.current;
        void enqueue(async () => {
            await release();
            if (version !== generation.current || !mounted.current) return;
            if (request === null || paused.current) {
                if (!blocked.current) setState(EMPTY_STATE);
                return;
            }
            if (blocked.current)
                throw new Error('Hardware release is unconfirmed. Restart Studio before reconnecting.');
            if (readyState !== ReadyState.OPEN) return;
            const id = uuidv4();
            active.current = { id, key: request.key };
            setState({ ...EMPTY_STATE, phase: 'connecting', key: request.key });
            sendJsonMessage({ command: 'connect', request_id: id, robot: request.robot }, false);
        }).catch((error: unknown) => fail(error instanceof Error ? error.message : 'Robot connection failed.'));
    }, [enqueue, fail, readyState, release, request, sendJsonMessage]);

    useEffect(() => {
        if (state.phase !== 'connecting') return;
        const timer = window.setTimeout(
            () => fail('Connecting is taking too long. Disconnect before retrying.'),
            CONNECTION_TIMEOUT_MS
        );
        return () => window.clearTimeout(timer);
    }, [fail, state.phase]);

    useEffect(() => {
        if (state.phase !== 'connected') return;
        const timer = window.setTimeout(
            () => fail('Live joint states stopped updating. Disconnect before retrying.'),
            3_000
        );
        return () => window.clearTimeout(timer);
    }, [fail, state.joints, state.phase]);

    useEffect(() => {
        mounted.current = true;
        return () => {
            mounted.current = false;
            // Closing the form socket triggers shielded backend cleanup, even if
            // a connect/read is still running. Never queue a new hardware command.
            releaseWaiter.current?.reject(new Error('Robot form closed.'));
        };
    }, []);

    const disconnect = useCallback(() => {
        paused.current = true;
        ++generation.current;
        return enqueue(release);
    }, [enqueue, release]);

    const retry = useCallback(() => {
        paused.current = false;
        if (readyState !== ReadyState.OPEN) setSocketAttempt((value) => value + 1);
        // New request identity reruns the lifecycle without infinite retries.
    }, [readyState]);

    return { ...state, disconnect, retry };
};
