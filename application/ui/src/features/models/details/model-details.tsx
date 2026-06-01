import { Suspense } from 'react';

import { Divider, Flex, Grid, Heading, Loading, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import type { components, SchemaModel } from '../../../api/openapi-spec';
import { Box } from '../../../routes/models/box.component';
import { formatDuration } from '../../../routes/models/utils';

interface ModelDetailsProps {
    model: SchemaModel;
}

type ExportDetail = components['schemas']['BackendExportDetail'];
type IOFeature = components['schemas']['IOFeature'];

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

const formatShape = (shape: IOFeature['shape']) => {
    if (shape === null || shape === undefined) {
        return '-';
    }

    return shape.length === 0 ? 'scalar' : `[${shape.join(', ')}]`;
};

const FeatureRow = ({ feature }: { feature: IOFeature }) => {
    return (
        <Flex gap='size-200' UNSAFE_style={{ padding: '4px 0', borderBottom: '1px solid var(--spectrum-gray-200)' }}>
            <Text UNSAFE_style={{ width: '260px', flexShrink: 0, fontWeight: 500 }}>{feature.name}</Text>
            <Text>
                {[feature.ftype, formatShape(feature.shape), feature.dtype].filter(Boolean).join(' / ')}
            </Text>
        </Flex>
    );
};

const NameList = ({ title, names }: { title: string; names: string[] | undefined }) => {
    if (names === undefined) {
        return null;
    }

    if (names.length === 0) {
        return null;
    }

    return <DetailRow name={title} value={names.join(', ')} />;
};

const IOSpec = ({ exports }: { exports: ExportDetail[] }) => {
    const exportsWithIoSpec = exports.filter(({ io_spec }) => io_spec !== null && io_spec !== undefined);

    if (exportsWithIoSpec.length === 0) {
        return null;
    }

    return (
        <View gridArea='io'>
            <Box
                title='I/O specification'
                content={
                    <Flex direction='column' gap='size-200'>
                        {exportsWithIoSpec.map((exportDetail) => {
                            const ioSpec = exportDetail.io_spec!;

                            return (
                                <Flex key={exportDetail.type} direction='column' gap='size-75'>
                                    <Heading level={4} marginBottom={0} marginTop={0}>
                                        {exportDetail.type}
                                    </Heading>

                                    {ioSpec.input_features?.length > 0 && (
                                        <Flex direction='column' gap='size-50'>
                                            <Text UNSAFE_style={{ fontWeight: 600 }}>Inputs</Text>
                                            {ioSpec.input_features.map((feature) => (
                                                <FeatureRow key={feature.name} feature={feature} />
                                            ))}
                                        </Flex>
                                    )}

                                    {ioSpec.output_features?.length > 0 && (
                                        <Flex direction='column' gap='size-50'>
                                            <Text UNSAFE_style={{ fontWeight: 600 }}>Outputs</Text>
                                            {ioSpec.output_features.map((feature) => (
                                                <FeatureRow key={feature.name} feature={feature} />
                                            ))}
                                        </Flex>
                                    )}

                                    <NameList title='Input names' names={ioSpec.input_names} />
                                    <NameList title='Output names' names={ioSpec.output_names} />
                                </Flex>
                            );
                        })}
                    </Flex>
                }
            />
        </View>
    );
};

const ModelDetailsContent = ({ model }: { model: SchemaModel }) => {
    const { data: modelDetail } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id: model.id! } },
    });

    const summary = modelDetail.training_summary;
    const hparams = modelDetail.hparams;

    return (
        <Grid areas={['model parameters', 'training parameters', 'io io']} gap='size-200'>
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

            <IOSpec exports={modelDetail.exports} />
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
