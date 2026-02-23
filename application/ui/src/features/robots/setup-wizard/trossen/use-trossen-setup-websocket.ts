import { useCallback, useRef, useState } from 'react';

import useWebSocket from 'react-use-websocket';

// ---------------------------------------------------------------------------
// Types — mirrors the backend TrossenSetupWorker protocol
// ---------------------------------------------------------------------------

export interface DiagnosticsResult {
    event: 'diagnostics_result';
    ip_reachable: boolean;
    configure_ok: boolean;
    motor_count: number;
    motor_names: string[];
    robot_type: string;
    connection_string: string;
    error_message: string | null;
}

export interface StatusEvent {
    event: 'status';
    state: string;
    phase: string;
    message: string;
}

export interface StateWasUpdatedEvent {
    event: 'state_was_updated';
    /** Joint positions keyed as "{motor_name}.pos" (degrees, gripper in meters) */
    state: Record<string, number>;
}

export interface ErrorEvent {
    event: 'error';
    message: string;
}

export type TrossenSetupEvent = DiagnosticsResult | StatusEvent | StateWasUpdatedEvent | ErrorEvent | { event: 'pong' };

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

interface UseTrossenSetupWebSocketOptions {
    projectId: string;
    robotType: string;
    connectionString: string;
    enabled?: boolean;
}

export interface TrossenSetupWebSocketState {
    /** Current backend phase */
    phase: string | null;
    /** Latest status message */
    statusMessage: string | null;
    /** Diagnostics result (IP ping + configure check) */
    diagnosticsResult: DiagnosticsResult | null;
    /** Normalized joint state for 3D preview (from state_was_updated events) */
    jointState: Record<string, number> | null;
    /** Latest error */
    error: string | null;
    /** Whether the websocket is connected */
    isConnected: boolean;
}

export function useTrossenSetupWebSocket({
    projectId,
    robotType,
    connectionString,
    enabled = true,
}: UseTrossenSetupWebSocketOptions) {
    const [state, setState] = useState<TrossenSetupWebSocketState>({
        phase: null,
        statusMessage: null,
        diagnosticsResult: null,
        jointState: null,
        error: null,
        isConnected: false,
    });

    const stateRef = useRef(state);
    stateRef.current = state;

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const data = JSON.parse(event.data) as TrossenSetupEvent;

            setState((prev) => {
                switch (data.event) {
                    case 'status':
                        return {
                            ...prev,
                            phase: data.phase,
                            statusMessage: data.message,
                            error: null,
                        };

                    case 'diagnostics_result':
                        return { ...prev, diagnosticsResult: data };

                    case 'state_was_updated':
                        return { ...prev, jointState: data.state };

                    case 'error':
                        return { ...prev, error: data.message };

                    case 'pong':
                        return prev;

                    default:
                        return prev;
                }
            });
        } catch (err) {
            console.error('Failed to parse Trossen setup websocket message:', err);
        }
    }, []);

    const url =
        enabled && robotType && connectionString
            ? `/api/projects/${projectId}/robots/setup/ws` +
              `?robot_type=${encodeURIComponent(robotType)}` +
              `&connection_string=${encodeURIComponent(connectionString)}`
            : null;

    const { sendJsonMessage, readyState } = useWebSocket(url, {
        onMessage: handleMessage,
        onOpen: () => setState((prev) => ({ ...prev, isConnected: true, error: null })),
        onClose: () => setState((prev) => ({ ...prev, isConnected: false })),
        onError: () => setState((prev) => ({ ...prev, error: 'WebSocket connection error' })),
        shouldReconnect: () => false, // Don't auto-reconnect — user should retry explicitly
    });

    // ------------------------------------------------------------------
    // Command senders
    // ------------------------------------------------------------------

    const reProbe = useCallback(() => {
        // Clear previous results so the UI shows a loading state while rechecking
        setState((prev) => ({ ...prev, diagnosticsResult: null, error: null }));
        sendJsonMessage({ command: 're_probe' });
    }, [sendJsonMessage]);

    const enterVerification = useCallback(() => {
        sendJsonMessage({ command: 'enter_verification' });
    }, [sendJsonMessage]);

    const ping = useCallback(() => {
        sendJsonMessage({ command: 'ping' });
    }, [sendJsonMessage]);

    return {
        state,
        readyState,
        commands: {
            reProbe,
            enterVerification,
            ping,
        },
    };
}
