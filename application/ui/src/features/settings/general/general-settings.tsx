import { Heading, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { HuggingFaceSettingsForm } from './huggingface-settings-form';
import { LoggerSettingsForm } from './logger-settings-form';
import { StreamingSettingsForm } from './streaming-settings-form';
import { TrainerSettingsForm } from './trainer-settings-form';

import classes from './general-settings.module.css';

export const GeneralSettings = () => {
    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');

    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Heading level={1}>General</Heading>
            <Text>Configure streaming, trainer, Hugging Face, and training-run logging defaults.</Text>
            <Text UNSAFE_className={classes.datasetPath}>Datasets directory: {settings.geti_action_dataset_path}</Text>
            <StreamingSettingsForm streaming={settings.streaming} />
            <TrainerSettingsForm trainer={settings.trainer} />
            <HuggingFaceSettingsForm huggingface={settings.huggingface} />
            <LoggerSettingsForm logger={settings.logger} />
        </View>
    );
};
