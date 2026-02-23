import { Dispatch, SetStateAction, useCallback, useRef, useState } from 'react';

import { ActionButton, Button, Checkbox, Flex, Heading, Slider, Switch, Text, TextField, View } from '@geti/ui';

import { DiagnosticsResult, TrossenSetupWebSocketState } from './use-trossen-setup-websocket';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper'];

const PHASE_OPTIONS = ['CONNECTING', 'DIAGNOSTICS', 'VERIFICATION'] as const;

const DEFAULT_STATE: TrossenSetupWebSocketState = {
    phase: null,
    statusMessage: null,
    diagnosticsResult: null,
    jointState: null,
    error: null,
    isConnected: false,
};

// ---------------------------------------------------------------------------
// Mock hook — drop-in replacement for useTrossenSetupWebSocket
// ---------------------------------------------------------------------------

export interface MockTrossenSetupWebSocketReturn {
    state: TrossenSetupWebSocketState;
    commands: {
        reProbe: () => void;
        enterVerification: () => void;
        ping: () => void;
    };
    /** Setter exposed to the debug panel */
    setMockState: Dispatch<SetStateAction<TrossenSetupWebSocketState>>;
}

export function useMockTrossenSetupWebSocket(): MockTrossenSetupWebSocketReturn {
    const [state, setState] = useState<TrossenSetupWebSocketState>(DEFAULT_STATE);

    const reProbe = useCallback(() => {
        console.info('[TrossenDebug] reProbe called');
        setState((prev) => ({ ...prev, diagnosticsResult: null, error: null }));
    }, []);

    const enterVerification = useCallback(() => {
        console.info('[TrossenDebug] enterVerification called');
        setState((prev) => ({
            ...prev,
            phase: 'VERIFICATION',
            statusMessage: 'Entering verification phase',
        }));
    }, []);

    const ping = useCallback(() => {
        console.info('[TrossenDebug] ping called');
    }, []);

    return { state, commands: { reProbe, enterVerification, ping }, setMockState: setState };
}

// ---------------------------------------------------------------------------
// Draggable floating panel
// ---------------------------------------------------------------------------

const PANEL_STYLE: React.CSSProperties = {
    position: 'fixed',
    bottom: 16,
    right: 16,
    zIndex: 99999,
    width: 380,
    maxHeight: '80vh',
    overflowY: 'auto',
    borderRadius: 8,
    boxShadow: '0 8px 32px rgba(0,0,0,0.35)',
    border: '1px solid var(--spectrum-global-color-gray-300)',
    backgroundColor: 'var(--spectrum-global-color-gray-100)',
    fontFamily: 'var(--spectrum-alias-font-family-default)',
    fontSize: 13,
};

const HEADER_STYLE: React.CSSProperties = {
    padding: '8px 12px',
    cursor: 'grab',
    backgroundColor: 'var(--spectrum-global-color-gray-200)',
    borderRadius: '8px 8px 0 0',
    userSelect: 'none',
    borderBottom: '1px solid var(--spectrum-global-color-gray-300)',
};

const SECTION_STYLE: React.CSSProperties = {
    padding: '8px 12px',
    borderBottom: '1px solid var(--spectrum-global-color-gray-200)',
};

// ---------------------------------------------------------------------------
// Section components
// ---------------------------------------------------------------------------

const ConnectionSection = ({
    state,
    setMockState,
}: {
    state: TrossenSetupWebSocketState;
    setMockState: Dispatch<SetStateAction<TrossenSetupWebSocketState>>;
}) => (
    <div style={SECTION_STYLE}>
        <Switch
            isSelected={state.isConnected}
            onChange={(value) => setMockState((prev) => ({ ...prev, isConnected: value }))}
        >
            WebSocket Connected
        </Switch>
    </div>
);

const DiagnosticsSection = ({
    state,
    setMockState,
}: {
    state: TrossenSetupWebSocketState;
    setMockState: Dispatch<SetStateAction<TrossenSetupWebSocketState>>;
}) => {
    const [ipReachable, setIpReachable] = useState(true);
    const [configureOk, setConfigureOk] = useState(true);
    const [motorCount, setMotorCount] = useState(7);

    const sendResult = () => {
        const result: DiagnosticsResult = {
            event: 'diagnostics_result',
            ip_reachable: ipReachable,
            configure_ok: configureOk,
            motor_count: motorCount,
            motor_names: MOTOR_NAMES.slice(0, motorCount),
            robot_type: 'trossen_widowx_ai',
            connection_string: '192.168.1.100',
            error_message: configureOk ? null : 'Simulated configuration failure',
        };
        setMockState((prev) => ({ ...prev, diagnosticsResult: result }));
    };

    const clearResult = () => {
        setMockState((prev) => ({ ...prev, diagnosticsResult: null }));
    };

    return (
        <div style={SECTION_STYLE}>
            <Flex direction='column' gap='size-100'>
                <Text UNSAFE_style={{ fontWeight: 600 }}>Diagnostics Result</Text>
                <Checkbox isSelected={ipReachable} onChange={setIpReachable}>
                    IP Reachable
                </Checkbox>
                <Checkbox isSelected={configureOk} onChange={setConfigureOk}>
                    Configure OK
                </Checkbox>
                <Slider
                    label='Motor count'
                    value={motorCount}
                    minValue={0}
                    maxValue={7}
                    step={1}
                    onChange={setMotorCount}
                    width='100%'
                />
                <Flex gap='size-100'>
                    <Button variant='accent' onPress={sendResult}>
                        Send Result
                    </Button>
                    <Button variant='secondary' onPress={clearResult}>
                        Clear (Loading)
                    </Button>
                </Flex>
                {state.diagnosticsResult && (
                    <Text UNSAFE_style={{ fontSize: 11, color: 'var(--spectrum-global-color-gray-600)' }}>
                        Active: ip={String(state.diagnosticsResult.ip_reachable)}, cfg=
                        {String(state.diagnosticsResult.configure_ok)}, motors={state.diagnosticsResult.motor_count}
                    </Text>
                )}
            </Flex>
        </div>
    );
};

const JointSection = ({ setMockState }: { setMockState: Dispatch<SetStateAction<TrossenSetupWebSocketState>> }) => {
    const [joints, setJoints] = useState<Record<string, number>>(() =>
        Object.fromEntries(MOTOR_NAMES.map((name) => [`${name}.pos`, name === 'gripper' ? 0.02 : 0]))
    );

    const updateJoint = (key: string, value: number) => {
        setJoints((prev) => {
            const next = { ...prev, [key]: value };
            setMockState((prevState) => ({ ...prevState, jointState: next }));
            return next;
        });
    };

    const randomize = () => {
        const next: Record<string, number> = {};
        for (const name of MOTOR_NAMES) {
            const key = `${name}.pos`;
            next[key] = name === 'gripper' ? Math.random() * 0.04 : Math.random() * 360 - 180;
        }
        setJoints(next);
        setMockState((prev) => ({ ...prev, jointState: next }));
    };

    const zero = () => {
        const next: Record<string, number> = {};
        for (const name of MOTOR_NAMES) {
            next[`${name}.pos`] = 0;
        }
        setJoints(next);
        setMockState((prev) => ({ ...prev, jointState: next }));
    };

    return (
        <div style={SECTION_STYLE}>
            <Flex direction='column' gap='size-75'>
                <Flex alignItems='center' justifyContent='space-between'>
                    <Text UNSAFE_style={{ fontWeight: 600 }}>Joint Positions</Text>
                    <Flex gap='size-75'>
                        <ActionButton isQuiet onPress={randomize}>
                            Random
                        </ActionButton>
                        <ActionButton isQuiet onPress={zero}>
                            Zero
                        </ActionButton>
                    </Flex>
                </Flex>
                {MOTOR_NAMES.map((name) => {
                    const key = `${name}.pos`;
                    const isGripper = name === 'gripper';
                    return (
                        <Slider
                            key={name}
                            label={name}
                            value={joints[key]}
                            minValue={isGripper ? 0 : -180}
                            maxValue={isGripper ? 0.04 : 180}
                            step={isGripper ? 0.001 : 1}
                            onChange={(v) => updateJoint(key, v)}
                            width='100%'
                        />
                    );
                })}
            </Flex>
        </div>
    );
};

const StatusSection = ({
    state,
    setMockState,
}: {
    state: TrossenSetupWebSocketState;
    setMockState: Dispatch<SetStateAction<TrossenSetupWebSocketState>>;
}) => {
    const [message, setMessage] = useState('Connecting to robot...');

    return (
        <div style={SECTION_STYLE}>
            <Flex direction='column' gap='size-100'>
                <Text UNSAFE_style={{ fontWeight: 600 }}>Status / Phase</Text>
                <Flex gap='size-75' wrap>
                    {PHASE_OPTIONS.map((phase) => (
                        <ActionButton
                            key={phase}
                            isQuiet={state.phase !== phase}
                            onPress={() => setMockState((prev) => ({ ...prev, phase, statusMessage: message }))}
                        >
                            {phase}
                        </ActionButton>
                    ))}
                </Flex>
                <TextField label='Status message' value={message} onChange={setMessage} width='100%' />
                <Button
                    variant='secondary'
                    onPress={() => setMockState((prev) => ({ ...prev, statusMessage: message }))}
                >
                    Send Status
                </Button>
                {state.phase && (
                    <Text UNSAFE_style={{ fontSize: 11, color: 'var(--spectrum-global-color-gray-600)' }}>
                        Current: phase={state.phase}, msg=&quot;{state.statusMessage}&quot;
                    </Text>
                )}
            </Flex>
        </div>
    );
};

const ErrorSection = ({
    state,
    setMockState,
}: {
    state: TrossenSetupWebSocketState;
    setMockState: Dispatch<SetStateAction<TrossenSetupWebSocketState>>;
}) => {
    const [errorMsg, setErrorMsg] = useState('Simulated connection error');

    return (
        <div style={SECTION_STYLE}>
            <Flex direction='column' gap='size-100'>
                <Text UNSAFE_style={{ fontWeight: 600 }}>Errors</Text>
                <TextField label='Error message' value={errorMsg} onChange={setErrorMsg} width='100%' />
                <Flex gap='size-100'>
                    <Button variant='negative' onPress={() => setMockState((prev) => ({ ...prev, error: errorMsg }))}>
                        Send Error
                    </Button>
                    <Button variant='secondary' onPress={() => setMockState((prev) => ({ ...prev, error: null }))}>
                        Clear Error
                    </Button>
                </Flex>
                {state.error && (
                    <Text UNSAFE_style={{ fontSize: 11, color: 'var(--spectrum-semantic-negative-color-default)' }}>
                        Active: &quot;{state.error}&quot;
                    </Text>
                )}
            </Flex>
        </div>
    );
};

// ---------------------------------------------------------------------------
// Main debug panel component
// ---------------------------------------------------------------------------

interface TrossenDebugPanelProps {
    state: TrossenSetupWebSocketState;
    setMockState: Dispatch<SetStateAction<TrossenSetupWebSocketState>>;
}

export const TrossenDebugPanel = ({ state, setMockState }: TrossenDebugPanelProps) => {
    const [collapsed, setCollapsed] = useState(false);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const dragRef = useRef<{ startX: number; startY: number; posX: number; posY: number } | null>(null);

    const handlePointerDown = useCallback(
        (e: React.PointerEvent) => {
            dragRef.current = { startX: e.clientX, startY: e.clientY, posX: position.x, posY: position.y };
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
        },
        [position]
    );

    const handlePointerMove = useCallback((e: React.PointerEvent) => {
        if (!dragRef.current) return;
        const dx = e.clientX - dragRef.current.startX;
        const dy = e.clientY - dragRef.current.startY;
        setPosition({ x: dragRef.current.posX + dx, y: dragRef.current.posY + dy });
    }, []);

    const handlePointerUp = useCallback(() => {
        dragRef.current = null;
    }, []);

    const panelStyle: React.CSSProperties = {
        ...PANEL_STYLE,
        transform: `translate(${position.x}px, ${position.y}px)`,
    };

    if (collapsed) {
        return (
            <div style={{ ...panelStyle, width: 'auto', maxHeight: 'none', overflow: 'visible' }}>
                <div style={{ padding: '6px 12px', cursor: 'pointer' }} onClick={() => setCollapsed(false)}>
                    <Text UNSAFE_style={{ fontWeight: 600, fontSize: 12 }}>Debug Panel</Text>
                </div>
            </div>
        );
    }

    return (
        <div style={panelStyle}>
            {/* Draggable header */}
            <div
                style={HEADER_STYLE}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
            >
                <Flex alignItems='center' justifyContent='space-between'>
                    <Heading level={5} margin={0}>
                        Trossen Debug
                    </Heading>
                    <ActionButton isQuiet onPress={() => setCollapsed(true)} aria-label='Collapse debug panel'>
                        <View UNSAFE_style={{ fontSize: 14, lineHeight: 1 }}>_</View>
                    </ActionButton>
                </Flex>
            </div>

            {/* Sections */}
            <ConnectionSection state={state} setMockState={setMockState} />
            <DiagnosticsSection state={state} setMockState={setMockState} />
            <JointSection setMockState={setMockState} />
            <StatusSection state={state} setMockState={setMockState} />
            <ErrorSection state={state} setMockState={setMockState} />
        </div>
    );
};
