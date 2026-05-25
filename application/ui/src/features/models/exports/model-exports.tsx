import { Suspense } from 'react';

import {
    Badge,
    Cell,
    Column,
    Content,
    ContextualHelp,
    Flex,
    Heading,
    Icon,
    Link,
    Loading,
    Row,
    TableBody,
    TableHeader,
    TableView,
    Text,
    View,
} from '@geti-ui/ui';
import { DownloadIcon } from '@geti-ui/ui/icons';

import { $api, fetchClient } from '../../../api/client';
import type { components, SchemaModel } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, type InferenceBackendConfig } from '../inference-backends';

interface ModelExportsProps {
    model: SchemaModel;
}

type BackendExportDetail = components['schemas']['BackendExportDetail'];

const InferenceBackendLogo = ({ backend, isAvailable }: { backend: InferenceBackendConfig; isAvailable: boolean }) => {
    const Logo = backend.logo;
    const unavailableStyle = isAvailable ? undefined : { opacity: 0.4 };

    return (
        <Flex alignItems='center' gap='size-200' UNSAFE_style={unavailableStyle}>
            <Flex>
                <Logo height={'30px'} width={'30px'} />
            </Flex>
            <Flex direction='column' gap='size-10' justifyContent={'center'}>
                <Heading level={4} marginBottom={0}>
                    {backend.label}
                </Heading>
                <Text
                    UNSAFE_style={{
                        fontSize: '11px',
                    }}
                >
                    {backend.description}
                </Text>
            </Flex>
        </Flex>
    );
};

const formatDate = (dateStr: string | null | undefined): string => {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
};

const formatSize = (bytes: number): string => {
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) {
        return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb.toFixed(1)} MB`;
};

interface DetailCellProps {
    exportDetail: BackendExportDetail | undefined;
    backend: InferenceBackendConfig;
}

const DetailCell = ({ backend, exportDetail }: DetailCellProps) => {
    if (exportDetail === undefined) {
        return (
            <Flex gap='size-100' alignItems={'center'}>
                <Badge variant={'negative'} UNSAFE_style={{ padding: 0, opacity: 0.9 }}>
                    Unavailable
                </Badge>
                <ContextualHelp variant='help'>
                    <Heading>Export missing</Heading>
                    <Content>
                        <Text>
                            This model does not include an exported model for {backend.label}. Try retraining the model
                            to restart the model export.
                        </Text>
                    </Content>
                </ContextualHelp>
            </Flex>
        );
    }

    return (
        <Flex direction='column' gap='size-10'>
            <Text>{formatSize(exportDetail.size_bytes)}</Text>
            {exportDetail.exported_at && (
                <Text
                    UNSAFE_style={{
                        fontSize: '11px',
                        color: 'var(--spectrum-gray-600)',
                    }}
                >
                    {formatDate(exportDetail.exported_at)}
                </Text>
            )}
        </Flex>
    );
};

const DownloadLink = ({ backend, modelId }: { modelId: string; backend: InferenceBackendConfig }) => {
    const downloadUrl = fetchClient.PATH('/api/models/{model_id}/exports/{backend}/download', {
        params: { path: { model_id: modelId, backend: backend.type } },
    });

    return (
        <Link
            href={downloadUrl}
            aria-label={`Download ${backend.label} export`}
            UNSAFE_style={{ color: 'inherit', display: 'inline-flex' }}
            target='_blank'
            rel='noopener noreferrer'
        >
            <Icon size='S'>
                <DownloadIcon />
            </Icon>
        </Link>
    );
};

const ModelExportsContents = ({ model }: { model: SchemaModel }) => {
    const { data: modelDetail } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id: model.id! } },
    });
    const { data: policyBackends } = $api.useSuspenseQuery('get', '/api/policies/backends');

    const backends: Array<string> = policyBackends[model.policy] ?? [];

    return (
        <View marginTop='size-200'>
            <TableView aria-label='Projects' density='spacious'>
                <TableHeader>
                    <Column key='format'>Format</Column>
                    <Column key='details'>Details</Column>
                    <Column key='actions' align='end'>
                        Download
                    </Column>
                </TableHeader>
                <TableBody>
                    {backends.map((backendType) => {
                        const exportDetail = modelDetail.exports.find(({ type }) => type === backendType);
                        const isAvailable = exportDetail !== undefined;
                        const backend = INFERENCE_BACKENDS[backendType];

                        return (
                            <Row key={backendType}>
                                <Cell>
                                    <InferenceBackendLogo backend={backend} isAvailable={isAvailable} />
                                </Cell>
                                <Cell>
                                    <DetailCell backend={backend} exportDetail={exportDetail} />
                                </Cell>
                                <Cell>{isAvailable && <DownloadLink backend={backend} modelId={model.id!} />}</Cell>
                            </Row>
                        );
                    })}
                </TableBody>
            </TableView>
        </View>
    );
};

export const ModelExports = ({ model }: ModelExportsProps) => {
    return (
        <Suspense fallback={<Loading mode='inline' size='M' marginY='size-400' />}>
            <ModelExportsContents model={model} />
        </Suspense>
    );
};
