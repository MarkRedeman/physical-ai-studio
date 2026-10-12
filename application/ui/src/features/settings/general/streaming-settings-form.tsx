import { useState } from 'react';

import { Item, Picker, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaSettingsUpdate, SchemaStreamingSettings } from '../../../api/openapi-spec';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

const DEFAULT_PROFILE_ID = 'default';

type StreamingSettingsFormProps = { streaming: SchemaStreamingSettings };

export const StreamingSettingsForm = ({ streaming }: StreamingSettingsFormProps) => {
    const patchMutation = useSettingsPatch();
    const { data: streamingProfiles } = $api.useSuspenseQuery('get', '/api/system/streaming-profiles');
    const [profileId, setProfileId] = useState(streaming.profile ?? DEFAULT_PROFILE_ID);
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);
    const profiles = [
        {
            id: DEFAULT_PROFILE_ID,
            name: 'Default',
            description: 'Automatically selects the first video encoder that works on this backend.',
        },
        ...streamingProfiles.profiles,
    ];
    const selectedProfile = profiles.find((profile) => profile.id === profileId);

    const save = () => {
        const body: SchemaSettingsUpdate = {
            streaming: { profile: profileId === DEFAULT_PROFILE_ID ? null : profileId },
        };
        patchMutation.mutate(
            { body },
            {
                onSuccess: () => {
                    setDirty(false);
                    setSaved(true);
                },
            }
        );
    };

    return (
        <SettingsSection
            title='Streaming'
            description='Video encoding defaults for new dataset recordings.'
            isDirty={dirty}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            <Picker
                label='Configuration'
                width='100%'
                selectedKey={profileId}
                items={profiles}
                onSelectionChange={(key) => {
                    setProfileId(String(key));
                    setDirty(true);
                    setSaved(false);
                }}
            >
                {(profile) => (
                    <Item key={profile.id} textValue={profile.name}>
                        <Text>{profile.name}</Text>
                        <Text slot='description'>{profile.description}</Text>
                    </Item>
                )}
            </Picker>
            {selectedProfile !== undefined && <Text>{selectedProfile.description}</Text>}
        </SettingsSection>
    );
};
