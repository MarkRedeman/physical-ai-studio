import { Suspense, useEffect, useRef, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    DropZone,
    FileTrigger,
    Flex,
    Heading,
    Item,
    Loading,
    Picker,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';

import { $api, fetchClient } from '../../../api/client';
import type {
    SchemaDatasetImportJob,
    SchemaDatasetImportJobPayload,
    SchemaImportStep,
    SchemaJobStatus,
} from '../../../api/openapi-spec';
import { DownloadProgressContent } from '../../../components/download-progress-content';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { isAbortError } from '../../utils/download';

const VALID_SOURCE_HINTS = ['auto', 'lerobot_v2', 'lerobot_v3'] as const;
type SourceHint = (typeof VALID_SOURCE_HINTS)[number];
type EnvironmentOption = { id: string; name: string };

type ImportPhase =
    | { step: 'editing'; errorMessage?: string }
    | { step: 'awaiting_detection' }
    | { step: 'detection_failed'; errorMessage: string }
    | { step: 'ready_to_finalize' }
    | { step: 'awaiting_import' };

interface DatasetFields {
    datasetName: string;
    defaultTask: string;
    environmentId: string | undefined;
}

interface DraftManifestSummary {
    source_type?: string;
    source_format_version?: string;
    suggested_name?: string;
    statistics?: {
        episode_count?: number;
        frame_count?: number;
    };
    dataset_schema?: {
        cameras?: Array<{
            name?: string;
            width?: number;
            height?: number;
            fps?: number;
        }>;
        robots?: Array<{
            name?: string;
            type?: string;
            joints?: string[];
        }>;
    };
    warnings?: string[];
    missing_fields?: string[];
}

/** Narrow the job union to the dataset import variant. */
const isDatasetImportJob = (
    job: { type: string; payload: unknown } | undefined
): job is SchemaDatasetImportJob => {
    return job?.type === 'dataset_import';
};

/** Type-safe accessor for the dataset import payload from any job response. */
const getImportPayload = (
    job: { type: string; payload: unknown } | undefined
): SchemaDatasetImportJobPayload | undefined => {
    if (isDatasetImportJob(job)) {
        return job.payload;
    }
    return undefined;
};

// ---------------------------------------------------------------------------
// useDatasetUpload — shared upload hook for ImportDatasetForm & DetectionFailedForm
// ---------------------------------------------------------------------------

interface UseDatasetUploadResult {
    /** Start the two-phase prepare + upload flow. */
    upload: (file: File, source: SourceHint) => Promise<string | undefined>;
    /** Abort any in-flight request. */
    abort: () => void;
    /** Upload progress percentage (null while indeterminate). */
    progress: number | null;
    /** Underlying mutation state for isPending / isError / error. */
    mutation: ReturnType<typeof useMutation<{ id: string }, Error, { file: File; source: string }>>;
}

const useDatasetUpload = (projectId: string): UseDatasetUploadResult => {
    const [progress, setProgress] = useState<number | null>(null);
    const abortRef = useRef<XMLHttpRequest | null>(null);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
        };
    }, []);

    const mutation = useMutation({
        mutationFn: async ({ file, source }: { file: File; source: string }) => {
            // Phase 1: prepare job using typed fetchClient (no XHR needed — tiny request)
            const { data: preparedJob, error } = await fetchClient.POST(
                '/api/projects/{project_id}/imports/datasets:prepare',
                {
                    params: { path: { project_id: projectId } },
                    body: { source_hint: source },
                    bodySerializer: (body) => {
                        const fd = new FormData();
                        fd.append('source_hint', (body as { source_hint: string }).source_hint);
                        return fd;
                    },
                }
            );

            if (error || !preparedJob) {
                throw new Error('Failed to prepare dataset import job');
            }

            const jobId = (preparedJob as { id?: string }).id;
            if (!jobId) {
                throw new Error('Failed to prepare import: missing job id');
            }

            // Phase 2: upload archive via XHR (needs onprogress for large files)
            const uploadPath = fetchClient.PATH(
                '/api/projects/{project_id}/imports/datasets/{job_id}:upload',
                { params: { path: { project_id: projectId, job_id: jobId } } }
            );

            return await new Promise<{ id: string }>((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                abortRef.current = xhr;
                xhr.open('PUT', uploadPath);
                xhr.responseType = 'json';

                xhr.upload.onprogress = (event) => {
                    if (event.lengthComputable && event.total > 0) {
                        setProgress(Math.round((event.loaded / event.total) * 100));
                    } else {
                        setProgress(null);
                    }
                };

                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        const uploaded = xhr.response as { id?: string } | null;
                        resolve({ id: uploaded?.id ?? jobId });
                    } else {
                        reject(new Error(`Failed to upload dataset archive: ${xhr.status}`));
                    }
                };

                xhr.onerror = () => reject(new Error('Failed to upload dataset archive'));
                xhr.onabort = () => reject(new DOMException('Upload aborted', 'AbortError'));

                const fd = new FormData();
                fd.append('archive', file);
                xhr.send(fd);
            });
        },
        onMutate: () => {
            setProgress(null);
        },
        onSettled: () => {
            abortRef.current = null;
        },
    });

    const upload = async (file: File, source: SourceHint): Promise<string | undefined> => {
        try {
            const job = await mutation.mutateAsync({ file, source });
            return job.id;
        } catch (error) {
            if (isAbortError(error)) {
                return undefined;
            }
            return undefined;
        }
    };

    const abort = () => {
        abortRef.current?.abort();
        mutation.reset();
        setProgress(null);
    };

    return { upload, abort, progress, mutation };
};

// ---------------------------------------------------------------------------
// ImportDatasetForm — file picker + upload trigger
// ---------------------------------------------------------------------------

interface ImportDatasetFormProps {
    project_id: string;
    onClose: () => void;
    sourceHint?: SourceHint;
    errorMessage?: string;
    onFileSelected: (file: File) => void;
    archive: File | null;
    onUploaded: (jobId: string) => void;
}

const ImportDatasetForm = ({
    project_id,
    onClose,
    sourceHint = 'auto',
    errorMessage,
    onFileSelected,
    archive,
    onUploaded,
}: ImportDatasetFormProps) => {
    const { upload, abort, progress, mutation } = useDatasetUpload(project_id);

    const startUpload = async (file: File) => {
        onFileSelected(file);
        const jobId = await upload(file, sourceHint);
        if (jobId) {
            onUploaded(jobId);
        }
    };

    const handleFileList = (files: FileList | null) => {
        const file = files?.[0] ?? null;
        if (file !== null) {
            void startUpload(file);
        }
    };

    const onCancel = () => {
        if (mutation.isPending) {
            abort();
            return;
        }
        onClose();
    };

    return (
        <>
            <Content>
                {errorMessage ? <Text>{errorMessage}</Text> : null}

                <DropZone
                    isFilled={archive !== null}
                    onDrop={async (e) => {
                        const fileItem = e.items.find((item) => item.kind === 'file');
                        if (fileItem?.kind !== 'file') {
                            return;
                        }
                        const file = await fileItem.getFile();
                        if (file.name.endsWith('.zip')) {
                            void startUpload(file);
                        }
                    }}
                >
                    <Flex direction='column' gap='size-100'>
                        {archive !== null ? (
                            <Text>{archive.name}</Text>
                        ) : (
                            <Text>Drop a .zip archive here or click to browse</Text>
                        )}
                        <View>
                            <FileTrigger acceptedFileTypes={['.zip']} onSelect={handleFileList}>
                                <Button variant='secondary' isDisabled={mutation.isPending}>
                                    {archive !== null ? 'Choose a different file' : 'Browse'}
                                </Button>
                            </FileTrigger>
                        </View>
                    </Flex>
                </DropZone>

                <DownloadProgressContent
                    isError={mutation.isError && !isAbortError(mutation.error)}
                    isPending={mutation.isPending}
                    progress={progress}
                    errorMessage='Failed to upload dataset archive. Please try again.'
                    preparingMessage='Uploading dataset archive...'
                />
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onCancel}>
                    {mutation.isPending ? 'Abort upload' : 'Cancel'}
                </Button>
            </ButtonGroup>
        </>
    );
};

// ---------------------------------------------------------------------------
// DatasetAnalysisInProgress — polls job until detection completes
// ---------------------------------------------------------------------------

const TERMINAL_STATUSES: SchemaJobStatus[] = ['failed', 'canceled'];
const IMPORTING_STEPS: SchemaImportStep[] = ['ready_to_commit', 'importing_resource'];

interface DatasetAnalysisInProgressProps {
    jobId: string;
    onReady: () => void;
    onImporting: () => void;
    onCompleted: (datasetId: string) => void;
    onDetectionFailed: (message: string) => void;
    onFailed: (message: string) => void;
    onClose: () => void;
}

const DatasetAnalysisInProgress = ({
    jobId,
    onReady,
    onImporting,
    onCompleted,
    onDetectionFailed,
    onFailed,
    onClose,
}: DatasetAnalysisInProgressProps) => {
    const importJobQuery = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        {
            params: { path: { job_id: jobId } },
        },
        {
            refetchInterval: 1000,
        }
    );

    const hasTransitionedRef = useRef(false);

    const job = importJobQuery.data;
    const payload = getImportPayload(job);

    if (!hasTransitionedRef.current && job && payload) {
        if (TERMINAL_STATUSES.includes(job.status as SchemaJobStatus)) {
            hasTransitionedRef.current = true;

            if (payload.step === ('detecting_source' satisfies SchemaImportStep)) {
                onDetectionFailed(
                    job.message ?? 'Could not automatically detect the dataset format. Please select a format manually.'
                );
            } else {
                onFailed(job.message ?? 'Import failed during processing.');
            }
        } else if (job.status === ('completed' satisfies SchemaJobStatus) && payload.result_dataset_id) {
            hasTransitionedRef.current = true;
            onCompleted(payload.result_dataset_id);
        } else if (payload.step === ('waiting_for_user_input' satisfies SchemaImportStep)) {
            hasTransitionedRef.current = true;
            onReady();
        } else if (
            IMPORTING_STEPS.includes(payload.step) ||
            job.status === ('running' satisfies SchemaJobStatus)
        ) {
            hasTransitionedRef.current = true;
            onImporting();
        }
    }

    return (
        <>
            <Content>
                <Text>Upload accepted. Waiting for server-side dataset detection...</Text>
                {importJobQuery.isError ? <Text>Failed to query import job status.</Text> : null}
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onClose}>
                    Close
                </Button>
                <Button variant='accent' isDisabled isPending>
                    Processing upload...
                </Button>
            </ButtonGroup>
        </>
    );
};

// ---------------------------------------------------------------------------
// DetectionFailedForm — retry with explicit source hint
// ---------------------------------------------------------------------------

const USER_SOURCE_HINTS = VALID_SOURCE_HINTS.filter((hint) => hint !== 'auto');

interface DetectionFailedFormProps {
    project_id: string;
    archive: File | null;
    errorMessage: string;
    onRetry: (jobId: string) => void;
    onClose: () => void;
}

const DetectionFailedForm = ({ project_id, archive, errorMessage, onRetry, onClose }: DetectionFailedFormProps) => {
    const [sourceHint, setSourceHint] = useState<SourceHint>(USER_SOURCE_HINTS[0]);
    const { upload, abort, progress, mutation } = useDatasetUpload(project_id);

    const onRetryUpload = async () => {
        if (!archive) {
            return;
        }
        const jobId = await upload(archive, sourceHint);
        if (jobId) {
            onRetry(jobId);
        }
    };

    const onCancel = () => {
        if (mutation.isPending) {
            abort();
            return;
        }
        onClose();
    };

    return (
        <>
            <Content>
                <Text>{errorMessage}</Text>

                {archive !== null ? <Text>Archive: {archive.name}</Text> : null}

                <Picker
                    items={USER_SOURCE_HINTS.map((value) => ({ id: value, name: value }))}
                    selectedKey={sourceHint}
                    label='Dataset format'
                    onSelectionChange={(value) =>
                        setSourceHint((value ?? USER_SOURCE_HINTS[0]).toString() as SourceHint)
                    }
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>

                <DownloadProgressContent
                    isError={mutation.isError && !isAbortError(mutation.error)}
                    isPending={mutation.isPending}
                    progress={progress}
                    errorMessage='Failed to upload dataset archive. Please try again.'
                    preparingMessage='Re-uploading dataset archive...'
                />
            </Content>

            <ButtonGroup>
                <Button variant='secondary' onPress={onCancel}>
                    {mutation.isPending ? 'Abort upload' : 'Cancel'}
                </Button>
                <Button
                    variant='accent'
                    onPress={onRetryUpload}
                    isPending={mutation.isPending}
                    isDisabled={archive === null}
                >
                    Retry with selected format
                </Button>
            </ButtonGroup>
        </>
    );
};

// ---------------------------------------------------------------------------
// ConfirmDatasetImport — finalize form
// ---------------------------------------------------------------------------

interface ConfirmDatasetImportProps {
    project_id: string;
    jobId: string;
    onClose: () => void;
    environments: EnvironmentOption[];
    fields: DatasetFields;
    onFieldsChange: (fields: DatasetFields) => void;
    onFinalized: () => void;
}

const ConfirmDatasetImport = ({
    project_id,
    jobId,
    onClose,
    environments,
    fields,
    onFieldsChange,
    onFinalized,
}: ConfirmDatasetImportProps) => {
    const importJobQuery = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        {
            params: { path: { job_id: jobId } },
        },
        {
            refetchInterval: 1000,
        }
    );
    const job = importJobQuery.data;
    const payload = getImportPayload(job);
    const draft = payload?.dataset_manifest_draft as DraftManifestSummary | undefined;
    const detectedFormat = draft?.source_type ?? 'unknown';
    const formatVersion = draft?.source_format_version;
    const cameras = draft?.dataset_schema?.cameras ?? [];
    const robots = draft?.dataset_schema?.robots ?? [];

    const finalizeMutation = $api.useMutation('post', '/api/projects/{project_id}/imports/datasets/{job_id}:finalize');
    const [finalizeError, setFinalizeError] = useState<string | undefined>(undefined);
    const canFinalize = fields.environmentId !== undefined && fields.datasetName.trim().length > 0;

    useEffect(() => {
        const suggestedName = draft?.suggested_name?.trim();
        if (suggestedName && fields.datasetName.trim().length === 0) {
            onFieldsChange({ ...fields, datasetName: suggestedName });
        }
    }, [draft?.suggested_name, fields, onFieldsChange]);

    const onFinalize = async () => {
        if (!canFinalize || fields.environmentId === undefined) {
            return;
        }

        try {
            setFinalizeError(undefined);
            await finalizeMutation.mutateAsync({
                params: {
                    path: {
                        project_id,
                        job_id: jobId,
                    },
                },
                body: {
                    dataset_name: fields.datasetName,
                    environment_id: fields.environmentId,
                    default_task: fields.defaultTask.length > 0 ? fields.defaultTask : undefined,
                },
            });

            onFinalized();
        } catch {
            setFinalizeError('Failed to finalize import. Please check job state and try again.');
        }
    };

    return (
        <>
            <Content>
                <Text>
                    Analysis complete. Review the detected metadata below, then provide a dataset name and environment
                    to finalize the import.
                </Text>
                {finalizeError ? <Text>{finalizeError}</Text> : null}

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
                    isRequired
                    width='100%'
                    label='Dataset name'
                    value={fields.datasetName}
                    onChange={(value) => onFieldsChange({ ...fields, datasetName: value })}
                />

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

// ---------------------------------------------------------------------------
// DatasetImportInProgress — polls until import completes
// ---------------------------------------------------------------------------

interface DatasetImportInProgressProps {
    jobId: string;
    onClose: () => void;
    onCompleted?: (datasetId: string) => void;
}

const DatasetImportInProgress = ({ jobId, onClose, onCompleted }: DatasetImportInProgressProps) => {
    const importJobQuery = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        {
            params: { path: { job_id: jobId } },
        },
        {
            refetchInterval: 1000,
        }
    );

    const hasCompletedRef = useRef(false);

    const job = importJobQuery.data;
    const payload = getImportPayload(job);

    if (!hasCompletedRef.current && payload) {
        if (job?.status === ('completed' satisfies SchemaJobStatus) && payload.result_dataset_id) {
            hasCompletedRef.current = true;
            onCompleted?.(payload.result_dataset_id);
        }
    }

    return (
        <>
            <Content>
                <Text>Waiting for import to complete...</Text>
                {importJobQuery.isError ? <Text>Failed to query import job status.</Text> : null}
                <Loading mode='inline' variant='intel' size='L' />
            </Content>

            <ButtonGroup>
                <Button variant='secondary' onPress={onClose}>
                    Close
                </Button>
                <Button variant='accent' isDisabled isPending>
                    Importing dataset...
                </Button>
            </ButtonGroup>
        </>
    );
};

// ---------------------------------------------------------------------------
// ImportDatasetDialog — orchestrator
// ---------------------------------------------------------------------------

interface ImportDatasetDialogProps {
    project_id: string;
    onClose: () => void;
    initialJobId?: string;
    onPendingJobDismissed?: (jobId: string) => void;
    onImportCompleted?: (datasetId: string) => void;
}

const ImportDatasetDialog = ({
    project_id,
    onClose,
    initialJobId,
    onPendingJobDismissed,
    onImportCompleted,
}: ImportDatasetDialogProps) => {
    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });

    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const [archive, setArchive] = useState<File | null>(null);
    const [fields, setFields] = useState<DatasetFields>({
        datasetName: '',
        defaultTask: '',
        environmentId: environments[0]?.id,
    });
    const [importJobId, setImportJobId] = useState<string | undefined>(initialJobId);
    const [importPhase, setImportPhase] = useState<ImportPhase>(
        initialJobId ? { step: 'awaiting_detection' } : { step: 'editing' }
    );

    const onFileSelected = (file: File) => {
        setArchive(file);
        if (fields.datasetName.trim().length === 0) {
            const suggestion = file.name.endsWith('.zip') ? file.name.slice(0, -4) : file.name;
            setFields((prev) => ({ ...prev, datasetName: suggestion }));
        }
    };

    const onDialogClose = () => {
        if (importJobId && importPhase.step !== 'editing') {
            onPendingJobDismissed?.(importJobId);
        }
        onClose();
    };

    const onImportDone = (datasetId: string) => {
        void queryClient.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}'] });
        void queryClient.invalidateQueries({ queryKey: ['get', '/api/jobs'] });
        onImportCompleted?.(datasetId);
        onDialogClose();
        navigate(paths.project.datasets.show({ project_id, dataset_id: datasetId }));
    };

    return (
        <Dialog>
            <Heading>Import dataset</Heading>
            <Divider />

            {importPhase.step === 'editing' && (
                <ImportDatasetForm
                    project_id={project_id}
                    onClose={onDialogClose}
                    errorMessage={importPhase.errorMessage}
                    onFileSelected={onFileSelected}
                    archive={archive}
                    onUploaded={(jobId) => {
                        setImportJobId(jobId);
                        setImportPhase({ step: 'awaiting_detection' });
                    }}
                />
            )}

            {importPhase.step === 'awaiting_detection' && importJobId && (
                <DatasetAnalysisInProgress
                    jobId={importJobId}
                    onReady={() => {
                        setImportPhase({ step: 'ready_to_finalize' });
                    }}
                    onImporting={() => {
                        setImportPhase({ step: 'awaiting_import' });
                    }}
                    onCompleted={onImportDone}
                    onDetectionFailed={(errorMessage) => {
                        setImportPhase({ step: 'detection_failed', errorMessage });
                    }}
                    onFailed={(errorMessage) => {
                        setImportPhase({ step: 'editing', errorMessage });
                    }}
                    onClose={onDialogClose}
                />
            )}

            {importPhase.step === 'detection_failed' && (
                <DetectionFailedForm
                    project_id={project_id}
                    archive={archive}
                    errorMessage={importPhase.errorMessage}
                    onRetry={(jobId) => {
                        setImportJobId(jobId);
                        setImportPhase({ step: 'awaiting_detection' });
                    }}
                    onClose={onDialogClose}
                />
            )}

            {importPhase.step === 'ready_to_finalize' && importJobId && (
                <ConfirmDatasetImport
                    project_id={project_id}
                    jobId={importJobId}
                    onClose={onClose}
                    environments={environments}
                    fields={fields}
                    onFieldsChange={setFields}
                    onFinalized={() => {
                        setImportPhase({ step: 'awaiting_import' });
                    }}
                />
            )}

            {importPhase.step === 'awaiting_import' && importJobId && (
                <DatasetImportInProgress jobId={importJobId} onClose={onDialogClose} onCompleted={onImportDone} />
            )}
        </Dialog>
    );
};

// ---------------------------------------------------------------------------
// DatasetImportButton — public export
// ---------------------------------------------------------------------------

interface DatasetImportButtonProps {
    existingJobId?: string;
    buttonLabel?: string;
    onPendingJobDismissed?: (jobId: string) => void;
    onImportCompleted?: (datasetId: string) => void;
}

export const DatasetImportButton = ({
    existingJobId,
    buttonLabel = 'Import dataset',
    onPendingJobDismissed,
    onImportCompleted,
}: DatasetImportButtonProps = {}) => {
    const { project_id } = useProjectId();

    return (
        <DialogTrigger>
            <Button variant='secondary' alignSelf={'center'}>
                <Text>{buttonLabel}</Text>
            </Button>

            {(close) => (
                <Suspense>
                    <ImportDatasetDialog
                        project_id={project_id}
                        initialJobId={existingJobId}
                        onPendingJobDismissed={onPendingJobDismissed}
                        onImportCompleted={onImportCompleted}
                        onClose={() => {
                            close();
                        }}
                    />
                </Suspense>
            )}
        </DialogTrigger>
    );
};
