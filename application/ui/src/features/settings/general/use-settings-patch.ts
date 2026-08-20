import { $api } from '../../../api/client';

/**
 * PATCH a partial update to /api/settings.
 *
 * The backend merges the provided groups/fields, so each section saves only
 * the values it owns; omitted groups (and omitted secret fields) are kept as-is.
 */
export const useSettingsPatch = () =>
    $api.useMutation('patch', '/api/settings', {
        meta: { invalidates: [['get', '/api/settings']] },
    });
