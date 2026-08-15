import { useState } from 'react';

import { toast } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { getApiErrorMessage, isResourceInUseError } from '../../api/errors';

export const usePluginsQuery = () => {
    return $api.useSuspenseQuery('get', '/api/plugins', {
        meta: { skipInvalidation: true },
    });
};

export const useInstallPluginMutation = () => {
    return $api.useMutation('post', '/api/plugins/{plugin_id}/install', {
        meta: { invalidates: [['get', '/api/plugins']] },
    });
};

export const useUninstallPluginMutation = () => {
    return $api.useMutation('post', '/api/plugins/{plugin_id}/uninstall', {
        meta: { invalidates: [['get', '/api/plugins']] },
    });
};

export const useRestartServerMutation = () => {
    return $api.useMutation('post', '/api/system/restart', {
        meta: { skipInvalidation: true },
    });
};

export const usePluginActions = () => {
    const installMutation = useInstallPluginMutation();
    const uninstallMutation = useUninstallPluginMutation();
    const [restartRequired, setRestartRequired] = useState(false);
    const [busyId, setBusyId] = useState<string | undefined>(undefined);

    const isBusy = busyId !== undefined;

    const install = async (pluginId: string) => {
        setBusyId(pluginId);
        try {
            await installMutation.mutateAsync({ params: { path: { plugin_id: pluginId } } });
            setRestartRequired(true);
            toast.positive('Plugin installed. Restart the server to activate it.');
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
            setRestartRequired(true);
            toast.positive('Plugin uninstalled. Restart the server to apply the change.');
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
