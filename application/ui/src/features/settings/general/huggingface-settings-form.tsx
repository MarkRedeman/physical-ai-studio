import { useState } from 'react';

import { SchemaHuggingFaceSettings, SchemaSettingsUpdate } from '../../../api/openapi-spec';
import { SecretChange, SecretField } from './secret-field';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

type HuggingFaceSettingsFormProps = { huggingface: SchemaHuggingFaceSettings };

export const HuggingFaceSettingsForm = ({ huggingface }: HuggingFaceSettingsFormProps) => {
    const patchMutation = useSettingsPatch();
    const [token, setToken] = useState<SecretChange>({ draft: '', remove: false });
    const [saved, setSaved] = useState(false);
    const dirty = token.draft !== '' || token.remove;

    const save = () => {
        const body: SchemaSettingsUpdate = {
            huggingface: {
                hf_token: token.remove ? null : token.draft === '' ? undefined : token.draft,
            },
        };
        patchMutation.mutate(
            { body },
            {
                onSuccess: () => {
                    setToken({ draft: '', remove: false });
                    setSaved(true);
                },
            }
        );
    };

    return (
        <SettingsSection
            title='Hugging Face'
            description='Token used to authenticate downloads of pretrained training assets.'
            isDirty={dirty}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            <SecretField label='Hugging Face token' isSet={huggingface.hf_token != null} onChange={setToken} />
        </SettingsSection>
    );
};
