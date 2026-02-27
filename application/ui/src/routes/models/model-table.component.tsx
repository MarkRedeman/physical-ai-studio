import { ActionButton, Flex, Grid, Item, Key, Link, Menu, MenuTrigger, Text, View } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { SchemaJob, SchemaModel } from '../../api/openapi-spec';
import { paths } from '../../router';
import { GRID_COLUMNS } from './constants';
import { durationBetween } from './utils';

import classes from './model-table.module.scss';

export const ModelHeader = () => {
    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelHeader}>
            <Text>Model name</Text>
            <Text>Trained</Text>
            <Text>Duration</Text>
            <Text>Architecture</Text>
            <div />
            <div />
        </Grid>
    );
};

export const ModelRow = ({
    model,
    trainingJob,
    onDelete,
    onRetrain,
}: {
    model: SchemaModel;
    trainingJob?: SchemaJob;
    onDelete: () => void;
    onRetrain: () => void;
}) => {
    const isHuggingFaceImport = model.properties?.source === 'huggingface';

    const onAction = (key: Key) => {
        const action = key.toString();
        if (action === 'delete') {
            onDelete();
        }
        if (action === 'retrain') {
            onRetrain();
        }
        if (action === 'download') {
            fetch(`/api/models/${model.id}/export`).then(async (res) => {
                if (!res.ok) return;
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `${model.name}.zip`;
                link.click();
                URL.revokeObjectURL(url);
            });
        }
    };

    const duration =
        trainingJob?.start_time && trainingJob?.end_time
            ? durationBetween(trainingJob.start_time, trainingJob.end_time)
            : null;

    const disabledKeys = isHuggingFaceImport ? ['retrain'] : [];

    const version = model.version ?? 1;

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelRow}>
            <Flex alignItems='center' gap='size-100'>
                <Text>{model.name}</Text>
                {version > 1 && (
                    <Text UNSAFE_style={{ color: 'var(--spectrum-gray-600)', fontSize: '0.85em' }}>v{version}</Text>
                )}
                {isHuggingFaceImport && (
                    <Text UNSAFE_style={{ color: 'var(--spectrum-gray-500)', fontSize: '0.8em', fontStyle: 'italic' }}>
                        HF
                    </Text>
                )}
            </Flex>
            <Text>{new Date(model.created_at!).toLocaleString()}</Text>
            <Text UNSAFE_className={duration ? undefined : classes.modelInfo}>{duration ?? '—'}</Text>
            <Text>{model.policy.toUpperCase()}</Text>
            <Link
                href={paths.project.models.inference({
                    project_id: model.project_id,
                    model_id: model.id!,
                })}
            >
                Run model
            </Link>
            <View>
                <MenuTrigger>
                    <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }} aria-label='options'>
                        <MoreMenu />
                    </ActionButton>
                    <Menu onAction={onAction} disabledKeys={disabledKeys}>
                        <Item key='retrain'>Retrain</Item>
                        <Item key='download'>Download</Item>
                        <Item key='delete'>Delete</Item>
                    </Menu>
                </MenuTrigger>
            </View>
        </Grid>
    );
};
