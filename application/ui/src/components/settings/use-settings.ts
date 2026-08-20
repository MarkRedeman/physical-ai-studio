import { $api } from '../../api/client';
import { SchemaUserSettingsResponse } from '../../api/openapi-spec';

export const useSettings = (): SchemaUserSettingsResponse => {
    const { data: userSettings } = $api.useSuspenseQuery('get', '/api/settings');
    return userSettings;
};
