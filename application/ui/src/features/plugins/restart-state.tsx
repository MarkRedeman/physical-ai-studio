import { createContext, ReactNode, useContext, useEffect, useMemo, useRef, useState } from 'react';

import { AlertDialog, DialogContainer, Flex, ProgressCircle, Text } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../api/client';
import { fetchClient } from '../../api/client';
import { SchemaTrainJob } from '../../api/openapi-spec';

type RestartStatus = 'idle' | 'requesting' | 'waiting_for_down' | 'waiting_for_up' | 'failed';

type RestartStateValue = {
    restartRequired: boolean;
    restartStatus: RestartStatus;
    restartPromptOpen: boolean;
    activeTrainingJobCount: number;
    hasActiveTrainingJobs: boolean;
    triggerRestartRequired: () => void;
    openRestartPrompt: () => void;
    closeRestartPrompt: () => void;
    restartServer: () => Promise<void>;
};

const HEALTH_POLL_INTERVAL_MS = 1500;
const MAX_HEALTH_POLLS = 120;
const HEALTHY_POLLS_REQUIRED = 2;
const MIN_RESTART_DIALOG_MS = 2000;

const RestartStateContext = createContext<RestartStateValue | null>(null);

export const RestartStateProvider = ({ children }: { children: ReactNode }) => {
    const queryClient = useQueryClient();

    const restartMutation = $api.useMutation('post', '/api/system/restart', {
        meta: { skipInvalidation: true },
    });
    const { data: jobs = [] } = $api.useQuery('get', '/api/jobs');

    const [restartRequired, setRestartRequired] = useState(false);
    const [restartStatus, setRestartStatus] = useState<RestartStatus>('idle');
    const [restartPromptOpen, setRestartPromptOpen] = useState(false);
    const [isPollingHealth, setIsPollingHealth] = useState(false);
    const [downObserved, setDownObserved] = useState(false);
    const [healthyPollCount, setHealthyPollCount] = useState(0);
    const pollAttemptsRef = useRef(0);
    const restartRequestedAtRef = useRef<number | null>(null);
    const completionScheduledRef = useRef(false);
    const completionTimeoutRef = useRef<number | null>(null);
    const suppressDismissRef = useRef(false);
    const isRestarting = restartStatus !== 'idle' && restartStatus !== 'failed';

    const activeTrainingJobCount = jobs.filter(
        (job): job is SchemaTrainJob => job.type === 'training' && (job.status === 'running' || job.status === 'pending')
    ).length;

    const triggerRestartRequired = () => {
        setRestartRequired(true);
        setRestartStatus('idle');
    };

    const openRestartPrompt = () => {
        suppressDismissRef.current = false;
        setRestartPromptOpen(true);
    };

    const closeRestartPrompt = () => {
        suppressDismissRef.current = false;
        setRestartPromptOpen(false);
    };

    const onDialogDismiss = () => {
        if (suppressDismissRef.current) {
            return;
        }
        closeRestartPrompt();
    };

    const restartServer = async () => {
        if (restartMutation.isPending || isPollingHealth) {
            return;
        }

        setRestartStatus('requesting');
        suppressDismissRef.current = true;
        setDownObserved(false);
        setHealthyPollCount(0);
        pollAttemptsRef.current = 0;
        restartRequestedAtRef.current = Date.now();
        completionScheduledRef.current = false;
        if (completionTimeoutRef.current !== null) {
            window.clearTimeout(completionTimeoutRef.current);
            completionTimeoutRef.current = null;
        }

        try {
            await restartMutation.mutateAsync({});
        } catch {
            // The server may restart before it sends a response.
        }

        setRestartStatus('waiting_for_down');
        setIsPollingHealth(true);
    };

    useEffect(() => {
        if (!isPollingHealth) {
            return;
        }

        let cancelled = false;

        const checkHealth = async () => {
            pollAttemptsRef.current += 1;
            if (pollAttemptsRef.current > MAX_HEALTH_POLLS) {
                if (!cancelled) {
                    setIsPollingHealth(false);
                    setRestartStatus('failed');
                    suppressDismissRef.current = false;
                }
                return;
            }

            let healthy = false;
            try {
                const { data, error } = await fetchClient.GET('/api/health');
                healthy = error === undefined && data?.status === 'healthy';
            } catch {
                healthy = false;
            }

            if (cancelled) {
                return;
            }

            if (healthy) {
                const nextHealthyPollCount = healthyPollCount + 1;
                setHealthyPollCount(nextHealthyPollCount);

                if (nextHealthyPollCount >= HEALTHY_POLLS_REQUIRED) {
                    if (completionScheduledRef.current) {
                        return;
                    }

                    completionScheduledRef.current = true;
                    const startedAt = restartRequestedAtRef.current ?? Date.now();
                    const elapsedMs = Date.now() - startedAt;
                    const remainingMs = Math.max(0, MIN_RESTART_DIALOG_MS - elapsedMs);

                    completionTimeoutRef.current = window.setTimeout(() => {
                        if (cancelled) {
                            return;
                        }
                        queryClient.clear();
                        setIsPollingHealth(false);
                        setRestartRequired(false);
                        setRestartStatus('idle');
                        suppressDismissRef.current = false;
                        completionScheduledRef.current = false;
                        completionTimeoutRef.current = null;
                    }, remainingMs);

                    return;
                }

                setRestartStatus(downObserved ? 'waiting_for_up' : 'waiting_for_down');
                return;
            }

            setDownObserved(true);
            setHealthyPollCount(0);
            setRestartStatus('waiting_for_up');
        };

        const interval = window.setInterval(() => {
            void checkHealth();
        }, HEALTH_POLL_INTERVAL_MS);

        void checkHealth();

        return () => {
            cancelled = true;
            window.clearInterval(interval);
        };
    }, [downObserved, healthyPollCount, isPollingHealth]);

    useEffect(() => {
        return () => {
            if (completionTimeoutRef.current !== null) {
                window.clearTimeout(completionTimeoutRef.current);
            }
        };
    }, []);

    const value = useMemo(
        () => ({
            restartRequired,
            restartStatus,
            restartPromptOpen,
            activeTrainingJobCount,
            hasActiveTrainingJobs: activeTrainingJobCount > 0,
            triggerRestartRequired,
            openRestartPrompt,
            closeRestartPrompt,
            restartServer,
        }),
        [activeTrainingJobCount, restartPromptOpen, restartRequired, restartStatus]
    );

    const progressMessageByStatus: Partial<Record<RestartStatus, string>> = {
        requesting: 'Sending restart request…',
        waiting_for_down: 'Waiting for server shutdown…',
        waiting_for_up: 'Waiting for server startup…',
        failed: 'Could not confirm restart from health checks. You can retry.',
    };

    return (
        <RestartStateContext.Provider value={value}>
            {children}
            {restartRequired && restartPromptOpen ? (
                <DialogContainer onDismiss={onDialogDismiss}>
                    <AlertDialog
                        title='Restart server now?'
                        variant='warning'
                        primaryActionLabel={isRestarting ? 'Restarting…' : restartStatus === 'failed' ? 'Retry restart' : 'Restart now'}
                        cancelLabel={isRestarting ? undefined : 'Later'}
                        onCancel={isRestarting ? undefined : closeRestartPrompt}
                        onPrimaryAction={restartServer}
                        isPrimaryActionDisabled={isRestarting}
                    >
                        <Flex direction='column' gap='size-150'>
                            <Text>Plugin changes require a server restart to become active.</Text>
                            {activeTrainingJobCount > 0 ? (
                                <Text>
                                    Restarting now will interrupt {activeTrainingJobCount} active training job
                                    {activeTrainingJobCount === 1 ? '' : 's'}.
                                </Text>
                            ) : null}
                            {isRestarting && progressMessageByStatus[restartStatus] ? (
                                <Flex alignItems='center' gap='size-100'>
                                    <ProgressCircle aria-label='Restarting server' isIndeterminate size='S' />
                                    <Text>{progressMessageByStatus[restartStatus]}</Text>
                                </Flex>
                            ) : null}
                            {!isRestarting && restartStatus === 'failed' && progressMessageByStatus[restartStatus] ? (
                                <Text>{progressMessageByStatus[restartStatus]}</Text>
                            ) : null}
                        </Flex>
                    </AlertDialog>
                </DialogContainer>
            ) : null}
        </RestartStateContext.Provider>
    );
};

export const useRestartState = () => {
    const context = useContext(RestartStateContext);
    if (context === null) {
        throw new Error('useRestartState must be used within RestartStateProvider');
    }
    return context;
};
