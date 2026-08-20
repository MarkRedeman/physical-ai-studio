import { useState } from 'react';

import { Flex, Switch, TextField, View } from '@geti-ui/ui';

import { SchemaLoggerSettings, SchemaSettingsUpdate } from '../../../api/openapi-spec';
import { SecretChange, SecretField } from './secret-field';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

import classes from './general-settings.module.css';

type LoggerSettingsFormProps = {
    logger: SchemaLoggerSettings;
};

type LoggerProvider = SchemaLoggerSettings['providers'][number];

export const LoggerSettingsForm = ({ logger }: LoggerSettingsFormProps) => {
    const patchMutation = useSettingsPatch();

    const [providers, setProviders] = useState<SchemaLoggerSettings['providers']>(logger.providers);
    const [wandbProject, setWandbProject] = useState(logger.wandb_project ?? '');
    const [wandbEntity, setWandbEntity] = useState(logger.wandb_entity ?? '');
    const [wandbApiKey, setWandbApiKey] = useState<SecretChange>({ draft: '', remove: false });
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);

    const markDirty = () => {
        setDirty(true);
        setSaved(false);
    };

    const csvEnabled = providers.includes('csv');
    const tensorboardEnabled = providers.includes('tensorboard');
    const wandbEnabled = providers.includes('wandb');

    const toggleProvider = (provider: LoggerProvider, enabled: boolean) => {
        setProviders((previous) =>
            enabled ? [...previous, provider] : previous.filter((value) => value !== provider)
        );
        markDirty();
    };

    const save = () => {
        const body: SchemaSettingsUpdate = {
            logger: {
                providers,
                wandb_project: wandbProject === '' ? null : wandbProject,
                wandb_entity: wandbEntity === '' ? null : wandbEntity,
                wandb_api_key: wandbApiKey.remove ? null : wandbApiKey.draft === '' ? undefined : wandbApiKey.draft,
            },
        };
        patchMutation.mutate(
            { body },
            {
                onSuccess: () => {
                    setWandbApiKey({ draft: '', remove: false });
                    setDirty(false);
                    setSaved(true);
                },
            }
        );
    };

    return (
        <SettingsSection
            title='Logging'
            description='Loggers enabled for training runs. Multiple providers run simultaneously.'
            isDirty={dirty}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            <Switch isSelected={csvEnabled} onChange={(selected) => toggleProvider('csv', selected)}>
                CSV logger
            </Switch>
            <Switch isSelected={tensorboardEnabled} onChange={(selected) => toggleProvider('tensorboard', selected)}>
                TensorBoard logger
            </Switch>

            <Flex direction='column' gap='size-200'>
                <Switch isSelected={wandbEnabled} onChange={(selected) => toggleProvider('wandb', selected)}>
                    Weights &amp; Biases
                </Switch>
                <View UNSAFE_className={classes.wandbSettings}>
                    <TextField
                        label='W&B project'
                        value={wandbProject}
                        isDisabled={!wandbEnabled}
                        onChange={(value) => {
                            setWandbProject(value);
                            markDirty();
                        }}
                        placeholder='Required when W&B is enabled'
                        width='100%'
                    />
                    <TextField
                        label='W&B entity'
                        value={wandbEntity}
                        isDisabled={!wandbEnabled}
                        onChange={(value) => {
                            setWandbEntity(value);
                            markDirty();
                        }}
                        width='100%'
                    />
                    <SecretField
                        label='W&B API key'
                        isSet={logger.wandb_api_key !== null && logger.wandb_api_key !== undefined}
                        isDisabled={!wandbEnabled}
                        onChange={(change) => {
                            setWandbApiKey(change);
                            markDirty();
                        }}
                    />
                </View>
            </Flex>
        </SettingsSection>
    );
};
