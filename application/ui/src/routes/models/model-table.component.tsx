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
    onViewLogs,
}: {
    model: SchemaModel;
    trainingJob?: SchemaJob;
    onDelete: () => void;
    onRetrain: () => void;
    onViewLogs?: () => void;
}) => {
    const trainJobId = model.train_job_id;

    const onAction = (key: Key) => {
        const action = key.toString();
        if (action === 'delete') {
            onDelete();
        }
        if (action === 'retrain') {
            onRetrain();
        }
        if (action === 'logs') {
            onViewLogs?.();
        }
        if (action === 'download') {
            const link = document.createElement('a');
            link.href = `/api/models/${model.id}:export`;
            link.download = `${model.name}.zip`;
            link.click();
        }
    };

    const duration =
        trainingJob?.start_time && trainingJob?.end_time
            ? durationBetween(trainingJob.start_time, trainingJob.end_time)
            : null;

    const disabledKeys: string[] = [];
    if (!trainJobId) disabledKeys.push('logs');

    const version = model.version ?? 1;

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelRow}>
            <Flex alignItems='center' gap='size-100'>
                <Text>{model.name}</Text>
                {version > 1 && (
                    <Text UNSAFE_style={{ color: 'var(--spectrum-gray-600)', fontSize: '0.85em' }}>v{version}</Text>
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
                        <Item key='logs'>Logs</Item>
                        <Item key='retrain'>Retrain</Item>
                        <Item key='download'>Download</Item>
                        <Item key='delete'>Delete</Item>
                    </Menu>
                </MenuTrigger>
            </View>
        </Grid>
    );
};
