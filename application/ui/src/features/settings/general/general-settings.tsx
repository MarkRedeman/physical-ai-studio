import { Heading, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { HuggingFaceSettingsForm } from './huggingface-settings-form';
import { LoggerSettingsForm } from './logger-settings-form';
import { StreamingSettingsForm } from './streaming-settings-form';
import { TrainerSettingsForm } from './trainer-settings-form';

export const GeneralSettings = () => {
    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');

    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Heading level={1}>General</Heading>
            <Text>Configure streaming, trainer, Hugging Face, and logging defaults.</Text>
            <HuggingFaceSettingsForm huggingface={settings.huggingface} />
            <StreamingSettingsForm streaming={settings.streaming} />
            <LoggerSettingsForm logger={settings.logger} />
            <TrainerSettingsForm trainer={settings.trainer} />
        </View>
    );
};
