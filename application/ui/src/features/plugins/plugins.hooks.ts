import { $api } from '../../api/client';

export const usePluginsQuery = () => {
    return $api.useSuspenseQuery('get', '/api/plugins', {
        meta: { skipInvalidation: true },
    });
};

export const useInstallPluginMutation = () => {
    return $api.useMutation('post', '/api/plugins/{plugin_id}/install', {
        meta: { skipInvalidation: true },
    });
};

export const useUninstallPluginMutation = () => {
    return $api.useMutation('post', '/api/plugins/{plugin_id}/uninstall', {
        meta: { skipInvalidation: true },
    });
};

export const useRestartServerMutation = () => {
    return $api.useMutation('post', '/api/system/restart', {
        meta: { skipInvalidation: true },
    });
};
