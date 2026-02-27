import { ActionButton, Flex, Grid, Item, Key, Link, Menu, MenuTrigger, Text, View } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { SchemaModel } from '../../api/openapi-spec';
import { paths } from '../../router';
import { GRID_COLUMNS } from './constants';

import classes from './model-table.module.scss';

export const ModelHeader = () => {
    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelHeader}>
            <Text>Model name</Text>
            <Text>Trained</Text>
            <Text>Architecture</Text>
            <div />
            <div />
        </Grid>
    );
};

export const ModelRow = ({
    model,
    onDelete,
    onRetrain,
    onViewLogs,
    onExport,
}: {
    model: SchemaModel;
    onDelete: () => void;
    onRetrain: () => void;
    onViewLogs?: () => void;
    onExport: () => void;
}) => {
    const trainJobId = model.train_job_id;
    const isHuggingFaceImport = model.properties?.source === 'huggingface';

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
            onExport();
        }
    };

    const disabledKeys = isHuggingFaceImport ? ['retrain'] : [];
    if (!trainJobId) disabledKeys.push('logs');

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
