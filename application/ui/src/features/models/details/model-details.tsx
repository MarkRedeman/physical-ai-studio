import { Suspense } from 'react';

import { Divider, Flex, Grid, Heading, Loading, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaModel } from '../../../api/openapi-spec';
import { Box } from '../../../routes/models/box.component';
import { formatDuration } from '../../../routes/models/utils';

interface ModelDetailsProps {
    model: SchemaModel;
}

const SKIP_HPARAMS_KEYS = new Set(['dataset_stats']);

const isPrimitive = (v: unknown): v is string | number | boolean | null => {
    return v === null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean';
};

const DetailRow = ({ name, value }: { name: string; value: unknown }) => {
    const display = isPrimitive(value) ? String(value) : JSON.stringify(value);
    return (
        <Flex gap='size-200' UNSAFE_style={{ padding: '4px 0', borderBottom: '1px solid var(--spectrum-gray-200)' }}>
            <Text UNSAFE_style={{ width: '260px', flexShrink: 0, fontWeight: 500 }}>{name}</Text>
            <Text>{display}</Text>
        </Flex>
    );
};

const ModelDetailsContent = ({ model }: { model: SchemaModel }) => {
    const { data: modelDetail } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id: model.id! } },
    });

    const summary = modelDetail.training_summary;
    const hparams = modelDetail.hparams;

    return (
        <Grid areas={['model parameters', 'training parameters']} gap='size-200'>
            <View gridArea='model'>
                <Box
                    title={'Model'}
                    content={
                        <Flex direction='column' gap='size-50'>
                            <DetailRow name='Name' value={modelDetail.model.name} />
                            <DetailRow name='Policy' value={modelDetail.model.policy} />
                        </Flex>
                    }
                />
            </View>

            {summary && (
                <View gridArea='training'>
                    <Box
                        title='Training configuration'
                        content={
                            <Flex direction='column' gap='size-50'>
                                <DetailRow name='Max steps' value={summary.max_steps} />
                                <DetailRow name='Batch size' value={summary.batch_size} />
                                {summary.precision && <DetailRow name='Precision' value={summary.precision} />}
                                {summary.compile_model !== null && summary.compile_model !== undefined && (
                                    <DetailRow name='Compiled' value={summary.compile_model ? 'Yes' : 'No'} />
                                )}
                                {summary.val_split !== null &&
                                    summary.val_split !== undefined &&
                                    summary.val_split > 0 && (
                                        <DetailRow name='Validation split' value={summary.val_split} />
                                    )}
                                {summary.auto_scale_batch_size != null && (
                                    <DetailRow
                                        name='Auto-scale batch size'
                                        value={summary.auto_scale_batch_size ? 'Yes' : 'No'}
                                    />
                                )}
                                <DetailRow name='Workers' value={summary.num_workers ?? '—'} />
                                {summary.device_type && <DetailRow name='Device' value={summary.device_type} />}
                                {summary.training_duration_seconds != null && (
                                    <DetailRow
                                        name='Training time'
                                        value={formatDuration(summary.training_duration_seconds)}
                                    />
                                )}

                                <Divider orientation='horizontal' size='S' marginY='size-100' />

                                {hparams &&
                                    Object.entries(hparams)
                                        .filter(([key]) => !SKIP_HPARAMS_KEYS.has(key))
                                        .map(([key, value]) => <DetailRow key={key} name={key} value={value} />)}
                            </Flex>
                        }
                    />
                </View>
            )}

            {hparams && (
                <View gridArea='parameters'>
                    <Box
                        title='Model Hyperparameters'
                        content={
                            <Flex direction='column' gap='size-50'>
                                {Object.entries(hparams)
                                    .filter(([key]) => !SKIP_HPARAMS_KEYS.has(key))
                                    .map(([key, value]) => (
                                        <DetailRow key={key} name={key} value={value} />
                                    ))}
                            </Flex>
                        }
                    />
                </View>
            )}
        </Grid>
    );
};

export const ModelDetails = ({ model }: ModelDetailsProps) => {
    return (
        <Suspense fallback={<Loading mode='inline' size='M' marginY='size-400' />}>
            <ModelDetailsContent model={model} />
        </Suspense>
    );
};
