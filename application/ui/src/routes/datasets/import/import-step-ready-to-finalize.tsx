import {
    Button,
    ButtonGroup,
    Content,
    Flex,
    Heading,
    InlineAlert,
    Item,
    Picker,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaDatasetImportJob } from '../../../api/openapi-spec';
import type { DraftManifestSummary, FinalizeFields } from './use-dataset-import-job-state';

interface ImportStepReadyToFinalizeProps {
    importJob: SchemaDatasetImportJob;
    project_id: string;
    onClose: () => void;
    fields: FinalizeFields;
    onFieldsChange: (fields: FinalizeFields) => void;
}

export const ImportStepReadyToFinalize = ({
    importJob,
    project_id,
    onClose,
    fields,
    onFieldsChange,
}: ImportStepReadyToFinalizeProps) => {
    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });

    const importPayload = importJob?.payload;
    const draft = (importPayload?.dataset_manifest_draft ?? undefined) as DraftManifestSummary | undefined;

    const detectedFormat = draft?.source_type ?? 'unknown';
    const formatVersion = draft?.source_format_version;
    const cameras = draft?.dataset_schema?.cameras ?? [];
    const robots = draft?.dataset_schema?.robots ?? [];

    const finalizeMutation = $api.useMutation('post', '/api/projects/{project_id}/imports/datasets/{job_id}:finalize', {
        meta: {
            invalidates: [['get', '/api/jobs/{job_id}', { params: { path: { job_id: importJob.id! } } }]],
        },
    });
    const canFinalize = fields.environmentId !== undefined;

    const onFinalize = () => {
        if (!canFinalize || fields.environmentId === undefined) {
            return;
        }

        finalizeMutation.mutate({
            params: {
                path: {
                    project_id,
                    job_id: importJob.id!,
                },
            },
            body: {
                environment_id: fields.environmentId,
                default_task: fields.defaultTask,
            },
        });
    };

    return (
        <>
            <Content>
                <Text>
                    Analysis complete. Review the detected metadata below, then provide environment and optional task to
                    finalize the import.
                </Text>
                {finalizeMutation.isError ? (
                    <InlineAlert variant='negative'>
                        <Heading>Finalize import failed</Heading>
                        <Content>Failed to finalize import. Please check job state and try again.</Content>
                    </InlineAlert>
                ) : null}

                {draft && (
                    <View backgroundColor='gray-50' borderColor='gray-200' borderWidth='thick' padding='size-200'>
                        <Flex direction='column' gap='size-150'>
                            <Heading level={4}>Detected dataset summary</Heading>

                            <Flex direction='row' wrap='wrap' gap='size-300'>
                                <Flex direction='column' gap='size-50'>
                                    <Text>
                                        <strong>Source format</strong>
                                    </Text>
                                    <Text>
                                        {detectedFormat}
                                        {formatVersion ? ` (${formatVersion})` : ''}
                                    </Text>
                                </Flex>

                                <Flex direction='column' gap='size-50'>
                                    <Text>
                                        <strong>Statistics</strong>
                                    </Text>
                                    <Text>Episodes: {draft.statistics?.episode_count ?? '—'}</Text>
                                    <Text>Frames: {draft.statistics?.frame_count ?? '—'}</Text>
                                </Flex>

                                <Flex direction='column' gap='size-50'>
                                    <Text>
                                        <strong>Schema overview</strong>
                                    </Text>
                                    <Text>Cameras: {cameras.length}</Text>
                                    <Text>Robots: {robots.length}</Text>
                                </Flex>
                            </Flex>

                            {cameras.length > 0 && (
                                <Flex direction='column' gap='size-50'>
                                    <Text>
                                        <strong>Cameras</strong>
                                    </Text>
                                    {cameras.map((camera, index) => (
                                        <Text key={`${camera.name ?? 'camera'}-${index}`}>
                                            {camera.name ?? `Camera ${index + 1}`}: {camera.width ?? '—'}×
                                            {camera.height ?? '—'} @ {camera.fps ?? '—'} FPS
                                        </Text>
                                    ))}
                                </Flex>
                            )}

                            {robots.length > 0 && (
                                <Flex direction='column' gap='size-50'>
                                    <Text>
                                        <strong>Robots</strong>
                                    </Text>
                                    {robots.map((robot, index) => (
                                        <Text key={`${robot.name ?? 'robot'}-${index}`}>
                                            {robot.name ?? `Robot ${index + 1}`} ({robot.type ?? 'unknown'}) — joints:{' '}
                                            {robot.joints?.length ?? 0}
                                        </Text>
                                    ))}
                                </Flex>
                            )}

                            {draft.warnings && draft.warnings.length > 0 && (
                                <Flex direction='column' gap='size-50'>
                                    <Text>
                                        <strong>Warnings</strong>
                                    </Text>
                                    {draft.warnings.map((warning, index) => (
                                        <Text key={`${warning}-${index}`}>• {warning}</Text>
                                    ))}
                                </Flex>
                            )}

                            {draft.missing_fields && draft.missing_fields.length > 0 && (
                                <Flex direction='column' gap='size-50'>
                                    <Text>
                                        <strong>Missing fields</strong>
                                    </Text>
                                    {draft.missing_fields.map((field, index) => (
                                        <Text key={`${field}-${index}`}>• {field}</Text>
                                    ))}
                                </Flex>
                            )}
                        </Flex>
                    </View>
                )}

                <Picker
                    items={environments}
                    selectedKey={fields.environmentId}
                    label='Environment'
                    onSelectionChange={(value) =>
                        onFieldsChange({
                            ...fields,
                            environmentId: value === null ? undefined : value.toString(),
                        })
                    }
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>

                <TextField
                    width='100%'
                    label='Task'
                    value={fields.defaultTask}
                    onChange={(value) => onFieldsChange({ ...fields, defaultTask: value })}
                />
            </Content>

            <ButtonGroup>
                <Button variant='secondary' onPress={onClose} isDisabled={finalizeMutation.isPending}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={onFinalize}
                    isPending={finalizeMutation.isPending}
                    isDisabled={!canFinalize}
                >
                    Finalize import
                </Button>
            </ButtonGroup>
        </>
    );
};
