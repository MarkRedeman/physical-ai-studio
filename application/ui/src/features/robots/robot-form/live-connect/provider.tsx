import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react';

import { $api } from '../../../../api/client';
import { getApiErrorMessage } from '../../../../api/errors';
import { useProjectId } from '../../../projects/use-project';
import { RobotPayload, useRobotForm } from '../provider';
import { validateRequiredUiFields } from '../robot-schema/schema-validation';
import { JsonSchema } from '../robot-schema/types';
import { LiveConnectHardwareBusyContext } from './hardware-busy';
import { useLiveConnection } from './use-live-connection';

export const configurationKey = (type: string | undefined, payload: RobotPayload) => JSON.stringify([type, payload]);
const EMPTY_JOINTS: Record<string, number> = {};

const useLiveConnectState = () => {
    const { project_id } = useProjectId();
    const { activeType, payload, draftId } = useRobotForm();
    const [consentType, setConsentType] = useState<string | null>(null);
    const [attempt, setAttempt] = useState(0);
    const [paused, setPaused] = useState(false);
    useEffect(() => {
        setConsentType(null);
    }, [activeType]);
    const key = configurationKey(activeType, payload);
    const [validation, setValidation] = useState<{
        key: string;
        payload: RobotPayload | null;
        error: string | null;
    } | null>(null);
    const schema = $api.useQuery(
        'get',
        '/api/robots/catalog/{robot_type}/schema',
        {
            params: { path: { robot_type: activeType ?? '' } },
        },
        { enabled: activeType !== undefined }
    );
    const { mutateAsync: validate } = $api.useMutation('post', '/api/robots/catalog/{robot_type}/validate', {
        meta: { skipInvalidation: true },
    });

    useEffect(() => {
        if (activeType === undefined || schema.data === undefined) return;
        let cancelled = false;
        const snapshot = JSON.parse(key)[1] as RobotPayload;
        const missing = validateRequiredUiFields(schema.data as JsonSchema, snapshot);
        const timer = window.setTimeout(async () => {
            if (missing.length !== 0) {
                setValidation({ key, payload: null, error: missing.join('; ') });
                return;
            }
            try {
                const result = await validate({ params: { path: { robot_type: activeType } }, body: snapshot });
                if (!cancelled) setValidation({ key, payload: result, error: null });
            } catch (error) {
                if (!cancelled)
                    setValidation({
                        key,
                        payload: null,
                        error: getApiErrorMessage(error) ?? 'Invalid robot configuration.',
                    });
            }
        }, 500);
        return () => {
            cancelled = true;
            window.clearTimeout(timer);
        };
    }, [activeType, key, schema.data, validate]);

    const validatedPayload = validation?.key === key ? validation.payload : null;
    const needsSetup =
        (activeType === 'SO101_Follower' || activeType === 'SO101_Leader') &&
        (validatedPayload?.calibration == null || Object.keys(validatedPayload.calibration as object).length === 0);
    const enabled = activeType !== undefined && consentType === activeType;
    const request = useMemo(() => {
        if (!enabled || paused || needsSetup || activeType === undefined || validatedPayload === null) return null;
        return {
            key: `${key}:${attempt}`,
            robot: { id: draftId, name: 'Robot setup', type: activeType, payload: validatedPayload },
        };
    }, [activeType, attempt, draftId, enabled, key, needsSetup, paused, validatedPayload]);
    const connection = useLiveConnection(project_id, request);
    const { retry: retryConnection, disconnect: disconnectRobot } = connection;
    const verified = enabled && !paused && connection.phase === 'connected' && connection.key === request?.key;

    const retry = useCallback(() => {
        retryConnection();
        setPaused(false);
        setAttempt((previous) => previous + 1);
    }, [retryConnection]);
    const disconnect = useCallback(() => {
        setPaused(true);
        return disconnectRobot();
    }, [disconnectRobot]);
    const setEnabled = useCallback(
        (value: boolean) => {
            setConsentType(value ? (activeType ?? null) : null);
            retry();
        },
        [activeType, retry]
    );
    const validationError = schema.isError
        ? 'Could not load the robot configuration schema.'
        : validation?.key === key
          ? validation.error
          : null;
    const controls = useMemo(
        () => ({
            phase: connection.phase,
            error: connection.error,
            enabled,
            verified,
            validatedPayload,
            validationError,
            needsSetup,
            setEnabled,
            retry,
            disconnect,
            resume: retry,
        }),
        [
            connection.phase,
            connection.error,
            enabled,
            verified,
            validatedPayload,
            validationError,
            needsSetup,
            setEnabled,
            retry,
            disconnect,
        ]
    );
    return { controls, joints: verified ? connection.joints : EMPTY_JOINTS };
};

const LiveConnectContext = createContext<ReturnType<typeof useLiveConnectState>['controls'] | null>(null);
const LiveJointsContext = createContext(EMPTY_JOINTS);

export const LiveConnectProvider = ({ children }: { children: ReactNode }) => {
    const value = useLiveConnectState();
    const { controls } = value;
    const hardwareBusy =
        controls.enabled ||
        controls.phase === 'releasing' ||
        controls.phase === 'connecting' ||
        controls.phase === 'connected';
    return (
        <LiveConnectHardwareBusyContext.Provider value={hardwareBusy}>
            <LiveConnectContext.Provider value={controls}>
                <LiveJointsContext.Provider value={value.joints}>{children}</LiveJointsContext.Provider>
            </LiveConnectContext.Provider>
        </LiveConnectHardwareBusyContext.Provider>
    );
};

export const useLiveJoints = () => useContext(LiveJointsContext);

export const useLiveConnect = () => {
    const context = useContext(LiveConnectContext);
    if (context === null) throw new Error('useLiveConnect must be used within LiveConnectProvider');
    return context;
};
