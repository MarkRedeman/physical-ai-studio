import { createContext, ReactNode, useContext, useEffect, useMemo, useRef, useState } from 'react';

import { AlertDialog, DialogContainer, Flex, Text } from '@geti-ui/ui';

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

const RestartStateContext = createContext<RestartStateValue | null>(null);

export const RestartStateProvider = ({ children }: { children: ReactNode }) => {
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

    const activeTrainingJobCount = jobs.filter(
        (job): job is SchemaTrainJob => job.type === 'training' && (job.status === 'running' || job.status === 'pending')
    ).length;

    const triggerRestartRequired = () => {
        setRestartRequired(true);
        setRestartStatus('idle');
    };

    const openRestartPrompt = () => {
        setRestartPromptOpen(true);
    };

    const closeRestartPrompt = () => {
        setRestartPromptOpen(false);
    };

    const restartServer = async () => {
        if (restartMutation.isPending || isPollingHealth) {
            return;
        }

        setRestartStatus('requesting');
        setRestartPromptOpen(false);
        setDownObserved(false);
        setHealthyPollCount(0);
        pollAttemptsRef.current = 0;

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
                    setIsPollingHealth(false);
                    setRestartRequired(false);
                    setRestartStatus('idle');
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

    return (
        <RestartStateContext.Provider value={value}>
            {children}
            {restartRequired && restartPromptOpen ? (
                <DialogContainer onDismiss={closeRestartPrompt}>
                    <AlertDialog
                        title='Restart server now?'
                        variant='warning'
                        primaryActionLabel='Restart now'
                        cancelLabel='Later'
                        onCancel={closeRestartPrompt}
                        onPrimaryAction={restartServer}
                        isPrimaryActionDisabled={restartStatus !== 'idle' && restartStatus !== 'failed'}
                    >
                        <Flex direction='column' gap='size-150'>
                            <Text>Plugin changes require a server restart to become active.</Text>
                            {activeTrainingJobCount > 0 ? (
                                <Text>
                                    Restarting now will interrupt {activeTrainingJobCount} active training job
                                    {activeTrainingJobCount === 1 ? '' : 's'}.
                                </Text>
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
