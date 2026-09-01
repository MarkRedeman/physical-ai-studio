import { useState } from 'react';

import { toast } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { getApiErrorMessage, isResourceInUseError } from '../../api/errors';
import { useRestartState } from '../system/restart-state';

export const usePluginsQuery = () => {
    return $api.useSuspenseQuery('get', '/api/plugins', {
        meta: { skipInvalidation: true },
    });
};

export const usePluginRestoreStatusQuery = () => {
    return $api.useQuery('get', '/api/plugins/restore-status', {
        staleTime: 30_000,
    });
};

export const useRestorePluginsMutation = () => {
    return $api.useMutation('post', '/api/plugins:restore', {
        meta: {
            invalidates: [
                ['get', '/api/plugins'],
                ['get', '/api/plugins/restore-status'],
            ],
        },
    });
};

export const useInstallPluginMutation = () => {
    return $api.useMutation('post', '/api/plugins/{plugin_id}', {
        meta: { invalidates: [['get', '/api/plugins']] },
    });
};

export const useUninstallPluginMutation = () => {
    return $api.useMutation('delete', '/api/plugins/{plugin_id}', {
        meta: { invalidates: [['get', '/api/plugins']] },
    });
};

export const usePluginActions = () => {
    const installMutation = useInstallPluginMutation();
    const uninstallMutation = useUninstallPluginMutation();
    const { restartRequired, triggerRestartRequired, openRestartPrompt } = useRestartState();
    const [busyId, setBusyId] = useState<string | undefined>(undefined);

    const isBusy = busyId !== undefined;

    const install = async (pluginId: string) => {
        setBusyId(pluginId);
        try {
            await installMutation.mutateAsync({ params: { path: { plugin_id: pluginId } } });
            triggerRestartRequired();
            openRestartPrompt();
        } catch (error) {
            toast.negative(getApiErrorMessage(error) ?? 'Failed to install the plugin.');
        } finally {
            setBusyId(undefined);
        }
    };

    const uninstall = async (pluginId: string) => {
        setBusyId(pluginId);
        try {
            await uninstallMutation.mutateAsync({ params: { path: { plugin_id: pluginId } } });
            triggerRestartRequired();
            openRestartPrompt();
        } catch (error) {
            if (isResourceInUseError(error)) {
                toast.info(getApiErrorMessage(error) ?? 'This plugin is in use and cannot be uninstalled.');
                return;
            }
            toast.negative(getApiErrorMessage(error) ?? 'Failed to uninstall the plugin.');
        } finally {
            setBusyId(undefined);
        }
    };

    return { isBusy, busyId, restartRequired, install, uninstall };
};

export const usePluginRestoreActions = () => {
    const restoreMutation = useRestorePluginsMutation();
    const { triggerRestartRequired, openRestartPrompt } = useRestartState();

    const restore = async () => {
        try {
            const result = await restoreMutation.mutateAsync({});
            if (result.restored_plugin_ids.length > 0) {
                triggerRestartRequired();
                openRestartPrompt();
            }
            if (result.failed_plugin_ids.length > 0 || result.unknown_plugin_ids.length > 0) {
                toast.info('Some recorded plugins could not be restored.');
            } else if (result.restored_plugin_ids.length > 0) {
                toast.positive('Plugins restored. Restart the server to activate them.');
            }
        } catch (error) {
            toast.negative(getApiErrorMessage(error) ?? 'Failed to restore plugins.');
        }
    };

    return { isRestoring: restoreMutation.isPending, restore };
};
