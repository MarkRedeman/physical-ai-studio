import { createContext, ReactNode, useContext, useEffect, useMemo, useRef, useState } from 'react';

import { $api, fetchClient } from '../../api/client';

type RestartStatus = 'idle' | 'requesting' | 'waiting_for_down' | 'waiting_for_up' | 'failed';

type RestartStateValue = {
    restartRequired: boolean;
    restartStatus: RestartStatus;
    triggerRestartRequired: () => void;
    restartServer: () => Promise<void>;
};

const HEALTH_POLL_INTERVAL_MS = 1500;
const MAX_HEALTH_POLLS = 120;
const HEALTHY_POLLS_REQUIRED = 2;

const RestartStateContext = createContext<RestartStateValue | null>(null);

const restartStore = {
    restartRequired: false,
    restartStatus: 'idle' as RestartStatus,
    subscribers: new Set<() => void>(),
};

const notifyRestartSubscribers = () => {
    restartStore.subscribers.forEach((subscriber) => subscriber());
};

export const RestartStateProvider = ({ children }: { children: ReactNode }) => {
    const restartMutation = $api.useMutation('post', '/api/system/restart', {
        meta: { skipInvalidation: true },
    });
    const [restartRequired, setRestartRequired] = useState(restartStore.restartRequired);
    const [restartStatus, setRestartStatus] = useState<RestartStatus>(restartStore.restartStatus);
    const [isPollingHealth, setIsPollingHealth] = useState(false);
    const [downObserved, setDownObserved] = useState(false);
    const [healthyPollCount, setHealthyPollCount] = useState(0);
    const pollAttemptsRef = useRef(0);

    useEffect(() => {
        const syncFromStore = () => {
            setRestartRequired(restartStore.restartRequired);
            setRestartStatus(restartStore.restartStatus);
        };

        restartStore.subscribers.add(syncFromStore);
        syncFromStore();

        return () => {
            restartStore.subscribers.delete(syncFromStore);
        };
    }, []);

    const updateStore = (updates: Partial<Pick<typeof restartStore, 'restartRequired' | 'restartStatus'>>) => {
        if (updates.restartRequired !== undefined) {
            restartStore.restartRequired = updates.restartRequired;
            setRestartRequired(updates.restartRequired);
        }
        if (updates.restartStatus !== undefined) {
            restartStore.restartStatus = updates.restartStatus;
            setRestartStatus(updates.restartStatus);
        }
        notifyRestartSubscribers();
    };

    const triggerRestartRequired = () => {
        updateStore({ restartRequired: true, restartStatus: 'idle' });
    };

    const restartServer = async () => {
        if (restartMutation.isPending || isPollingHealth) {
            return;
        }

        updateStore({ restartStatus: 'requesting' });
        setDownObserved(false);
        setHealthyPollCount(0);
        pollAttemptsRef.current = 0;

        try {
            await restartMutation.mutateAsync({});
        } catch {
            // The server may restart before it sends a response.
        }

        updateStore({ restartStatus: 'waiting_for_down' });
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
                    updateStore({ restartStatus: 'failed' });
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
                    updateStore({ restartRequired: false, restartStatus: 'idle' });
                    return;
                }

                updateStore({ restartStatus: downObserved ? 'waiting_for_up' : 'waiting_for_down' });
                return;
            }

            setDownObserved(true);
            setHealthyPollCount(0);
            updateStore({ restartStatus: 'waiting_for_up' });
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
        () => ({ restartRequired, restartStatus, triggerRestartRequired, restartServer }),
        [restartRequired, restartStatus]
    );

    return <RestartStateContext.Provider value={value}>{children}</RestartStateContext.Provider>;
};

export const useRestartState = () => {
    const context = useContext(RestartStateContext);
    if (context === null) {
        throw new Error('useRestartState must be used within RestartStateProvider');
    }
    return context;
};
