import { Suspense, Suspense, useEffect, useRef, useState } from 'react';

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
    Picker,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';

import { $api, fetchClient } from '../../../api/client';
import { DownloadProgressContent } from '../../../components/download-progress-content';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { isAbortError } from '../../utils/download';

const VALID_SOURCE_HINTS = ['auto', 'studio', 'lerobot_v2', 'lerobot_v3', 'trossen_sdk'] as const;
type ImportPhase = 'editing' | 'awaiting_detection' | 'ready_to_finalize' | 'awaiting_import';
type EnvironmentOption = { id: string; name: string };

type ImportPayload = {
    step?: string;
    result_dataset_id?: string;
};

const asImportPayload = (payload: unknown): ImportPayload => {
    if (payload && typeof payload === 'object') {
        return payload as ImportPayload;
    }
    return {};
};

interface ImportDatasetFormProps {
    project_id: string;
    onClose: () => void;
    sourceHint: (typeof VALID_SOURCE_HINTS)[number];
    onSourceHintChange: (value: (typeof VALID_SOURCE_HINTS)[number]) => void;
    onFileSelect: (files: FileList | null) => void;
    archive: File | null;
    onUploaded: (jobId: string, message: string) => void;
}

const ImportDatasetForm = ({
    project_id,
    onClose,
    sourceHint,
    onSourceHintChange,
    onFileSelect,
    archive,
    onUploaded,
}: ImportDatasetFormProps) => {
    const [uploadProgress, setUploadProgress] = useState<number | null>(null);
    const abortRef = useRef<XMLHttpRequest | null>(null);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
        };
    }, []);

    const uploadPath = fetchClient.PATH('/api/projects/{project_id}/imports/datasets', {
        params: { path: { project_id } },
    });

    const uploadMutation = useMutation({
        mutationFn: async ({ file, source }: { file: File; source: string }) => {
            return await new Promise<{ id: string }>((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                abortRef.current = xhr;
                xhr.open('POST', uploadPath);
                xhr.responseType = 'json';

                xhr.upload.onprogress = (event) => {
                    if (event.lengthComputable && event.total > 0) {
                        const percent = Math.round((event.loaded / event.total) * 100);
                        setUploadProgress(percent);
                    } else {
                        setUploadProgress(null);
                    }
                };

                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(xhr.response as { id: string });
                    } else {
                        reject(new Error(`Failed to submit import: ${xhr.status}`));
                    }
                };

                xhr.onerror = () => reject(new Error('Failed to upload dataset archive'));
                xhr.onabort = () => reject(new DOMException('Upload aborted', 'AbortError'));

                const formData = new FormData();
                formData.append('archive', file);
                formData.append('source_hint', source);
                xhr.send(formData);
            });
        },
        onMutate: () => {
            setUploadProgress(null);
        },
        onSettled: () => {
            abortRef.current = null;
        },
    });
    const canUpload = archive !== null;

    const onUpload = async () => {
        if (!canUpload || archive === null) {
            return;
        }

        try {
            const job = await uploadMutation.mutateAsync({ file: archive, source: sourceHint });

            onUploaded(job.id, 'Upload accepted. Waiting for server-side dataset detection…');
        } catch (error) {
            if (isAbortError(error)) {
                return;
            }
        }
    };

    const onCancel = () => {
        if (uploadMutation.isPending) {
            abortRef.current?.abort();
            uploadMutation.reset();
            setUploadProgress(null);
            return;
        }
        onClose();
    };

    return (
        <>
            <Content>
                <DropZone
                    isFilled={archive !== null}
                    onDrop={async (e) => {
                        const fileItem = e.items.find((item) => item.kind === 'file');
                        if (fileItem?.kind !== 'file') {
                            return;
                        }
                        const file = await fileItem.getFile();
                        if (file.name.endsWith('.zip')) {
                            const dataTransfer = new DataTransfer();
                            dataTransfer.items.add(file);
                            onFileSelect(dataTransfer.files);
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
                            <FileTrigger acceptedFileTypes={['.zip']} onSelect={onFileSelect}>
                                <Button variant='secondary'>
                                    {archive !== null ? 'Choose a different file' : 'Browse'}
                                </Button>
                            </FileTrigger>
                        </View>
                    </Flex>
                </DropZone>

                <DownloadProgressContent
                    isError={uploadMutation.isError && !isAbortError(uploadMutation.error)}
                    isPending={uploadMutation.isPending}
                    progress={uploadProgress}
                    errorMessage='Failed to upload dataset archive. Please try again.'
                    preparingMessage='Uploading dataset archive…'
                />

                <Picker
                    items={VALID_SOURCE_HINTS.map((value) => ({ id: value, name: value }))}
                    selectedKey={sourceHint}
                    label='Source hint'
                    onSelectionChange={(value) =>
                        onSourceHintChange((value ?? 'auto').toString() as (typeof VALID_SOURCE_HINTS)[number])
                    }
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onCancel}>
                    {uploadMutation.isPending ? 'Abort upload' : 'Cancel'}
                </Button>
                <Button
                    variant='accent'
                    onPress={onUpload}
                    isPending={uploadMutation.isPending}
                    isDisabled={!canUpload}
                >
                    Upload dataset
                </Button>
            </ButtonGroup>
        </>
    );
};

interface DatasetAnalysisInProgressProps {
    jobId: string;
    statusMessage?: string;
    onReady: (message: string) => void;
    onImporting: (message: string) => void;
    onCompleted: (datasetId: string) => void;
    onFailed: (message: string) => void;
    onClose: () => void;
}

const DatasetAnalysisInProgress = ({
    jobId,
    statusMessage,
    onReady,
    onImporting,
    onCompleted,
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

    useEffect(() => {
        if (hasTransitionedRef.current) {
            return;
        }

        const job = importJobQuery.data;
        if (!job) {
            return;
        }

        const payload = asImportPayload(job.payload);

        if (job.status === 'failed' || job.status === 'canceled') {
            hasTransitionedRef.current = true;
            onFailed(job.message ?? 'Import failed during processing.');
            return;
        }

        if (job.status === 'completed' && payload.result_dataset_id) {
            hasTransitionedRef.current = true;
            onCompleted(payload.result_dataset_id);
            return;
        }

        if (payload.step === 'waiting_for_user_input') {
            hasTransitionedRef.current = true;
            onReady('Dataset analyzed. Review fields and click "Finalize import".');
            return;
        }

        if (payload.step === 'ready_to_commit' || payload.step === 'importing_resource' || job.status === 'running') {
            hasTransitionedRef.current = true;
            onImporting('Import is already running. Showing live import progress…');
        }
    }, [importJobQuery.data, onCompleted, onFailed, onImporting, onReady]);

    return (
        <>
            <Content>
                <Text>{statusMessage ?? 'Upload accepted. Waiting for server-side dataset detection…'}</Text>
                {importJobQuery.isError ? <Text>Failed to query import job status.</Text> : null}
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onClose}>
                    Close
                </Button>
                <Button variant='accent' isDisabled isPending>
                    Processing upload…
                </Button>
            </ButtonGroup>
        </>
    );
};

interface ConfirmDatasetImportProps {
    project_id: string;
    jobId: string;
    onClose: () => void;
    environments: EnvironmentOption[];
    environmentId: string | undefined;
    onEnvironmentChange: (value: string | undefined) => void;
    datasetName: string;
    onDatasetNameChange: (value: string) => void;
    defaultTask: string;
    onDefaultTaskChange: (value: string) => void;
    statusMessage?: string;
    onFinalized: (message: string) => void;
}

const ConfirmDatasetImport = ({
    project_id,
    jobId,
    onClose,
    environments,
    environmentId,
    onEnvironmentChange,
    datasetName,
    onDatasetNameChange,
    defaultTask,
    onDefaultTaskChange,
    statusMessage,
    onFinalized,
}: ConfirmDatasetImportProps) => {
    const finalizeMutation = $api.useMutation('post', '/api/projects/{project_id}/imports/datasets/{job_id}:finalize');
    const [finalizeError, setFinalizeError] = useState<string | undefined>(undefined);
    const canFinalize = environmentId !== undefined && datasetName.trim().length > 0;

    const onFinalize = async () => {
        if (!canFinalize || environmentId === undefined) {
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
                    dataset_name: datasetName,
                    environment_id: environmentId,
                    default_task: defaultTask.length > 0 ? defaultTask : undefined,
                },
            });

            onFinalized('Finalize accepted. Waiting for import to complete…');
        } catch {
            setFinalizeError('Failed to finalize import. Please check job state and try again.');
        }
    };

    return (
        <>
            <Content>
                <Text>{statusMessage ?? 'Dataset analyzed. Review fields and click "Finalize import".'}</Text>
                {finalizeError ? <Text>{finalizeError}</Text> : null}

                <Picker
                    items={environments}
                    selectedKey={environmentId}
                    label='Environment'
                    onSelectionChange={(value) => onEnvironmentChange(value === null ? undefined : value.toString())}
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>

                <TextField
                    isRequired
                    width='100%'
                    label='Dataset name'
                    value={datasetName}
                    onChange={onDatasetNameChange}
                />

                <TextField width='100%' label='Task' value={defaultTask} onChange={onDefaultTaskChange} />
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

interface DatasetImportInProgressProps {
    project_id: string;
    jobId: string;
    statusMessage?: string;
    onClose: () => void;
    onCompleted?: (datasetId: string) => void;
}

const DatasetImportInProgress = ({
    project_id,
    jobId,
    statusMessage,
    onClose,
    onCompleted,
}: DatasetImportInProgressProps) => {
    const navigate = useNavigate();
    const queryClient = useQueryClient();
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

    useEffect(() => {
        const job = importJobQuery.data;
        if (!job) {
            return;
        }

        const payload = asImportPayload(job.payload);

        if (job.status === 'completed' && payload.result_dataset_id) {
            void queryClient.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}'] });
            onCompleted?.(payload.result_dataset_id);
            onClose();
            navigate(paths.project.datasets.show({ project_id, dataset_id: payload.result_dataset_id }));
        }
    }, [importJobQuery.data, navigate, onClose, onCompleted, project_id, queryClient]);

    return (
        <>
            <Content>
                <Text>{statusMessage ?? 'Finalize accepted. Waiting for import to complete…'}</Text>
                {importJobQuery.isError ? <Text>Failed to query import job status.</Text> : null}
            </Content>

            <ButtonGroup>
                <Button variant='secondary' onPress={onClose}>
                    Close
                </Button>
                <Button variant='accent' isDisabled isPending>
                    Importing dataset…
                </Button>
            </ButtonGroup>
        </>
    );
};

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
    const [archive, setArchive] = useState<File | null>(null);
    const [datasetName, setDatasetName] = useState('');
    const [defaultTask, setDefaultTask] = useState('');
    const [environmentId, setEnvironmentId] = useState<string | undefined>(() => environments[0]?.id);
    const [sourceHint, setSourceHint] = useState<(typeof VALID_SOURCE_HINTS)[number]>('auto');
    const [importJobId, setImportJobId] = useState<string | undefined>(initialJobId);
    const [importPhase, setImportPhase] = useState<ImportPhase>(initialJobId ? 'awaiting_detection' : 'editing');
    const [importStatusMessage, setImportStatusMessage] = useState<string | undefined>(undefined);

    const onFileSelect = (files: FileList | null) => {
        const file = files?.[0] ?? null;
        setArchive(file);
        if (file !== null && datasetName.trim().length === 0) {
            const suggestion = file.name.endsWith('.zip') ? file.name.slice(0, -4) : file.name;
            setDatasetName(suggestion);
        }
    };

    const onDialogClose = () => {
        if (importJobId && importPhase !== 'editing') {
            onPendingJobDismissed?.(importJobId);
        }
        onClose();
    };

    return (
        <Dialog>
            <Heading>Import dataset</Heading>
            <Divider />

            {importPhase === 'editing' && (
                <ImportDatasetForm
                    project_id={project_id}
                    onClose={onDialogClose}
                    sourceHint={sourceHint}
                    onSourceHintChange={setSourceHint}
                    onFileSelect={onFileSelect}
                    archive={archive}
                    onUploaded={(jobId, message) => {
                        setImportJobId(jobId);
                        setImportStatusMessage(message);
                        setImportPhase('awaiting_detection');
                    }}
                />
            )}

            {importPhase === 'awaiting_detection' && importJobId && (
                <DatasetAnalysisInProgress
                    jobId={importJobId}
                    statusMessage={importStatusMessage}
                    onReady={(message) => {
                        setImportStatusMessage(message);
                        setImportPhase('ready_to_finalize');
                    }}
                    onImporting={(message) => {
                        setImportStatusMessage(message);
                        setImportPhase('awaiting_import');
                    }}
                    onCompleted={(datasetId) => {
                        onImportCompleted?.(datasetId);
                        onDialogClose();
                        navigate(paths.project.datasets.show({ project_id, dataset_id: datasetId }));
                    }}
                    onFailed={(message) => {
                        setImportStatusMessage(message);
                        setImportPhase('editing');
                    }}
                    onClose={onDialogClose}
                />
            )}

            {importPhase === 'ready_to_finalize' && importJobId && (
                <ConfirmDatasetImport
                    project_id={project_id}
                    jobId={importJobId}
                    onClose={onClose}
                    environments={environments}
                    environmentId={environmentId}
                    onEnvironmentChange={setEnvironmentId}
                    datasetName={datasetName}
                    onDatasetNameChange={setDatasetName}
                    defaultTask={defaultTask}
                    onDefaultTaskChange={setDefaultTask}
                    statusMessage={importStatusMessage}
                    onFinalized={(message) => {
                        setImportStatusMessage(message);
                        setImportPhase('awaiting_import');
                    }}
                />
            )}

            {importPhase === 'awaiting_import' && importJobId && (
                <DatasetImportInProgress
                    project_id={project_id}
                    jobId={importJobId}
                    statusMessage={importStatusMessage}
                    onClose={onDialogClose}
                    onCompleted={onImportCompleted}
                />
            )}
        </Dialog>
    );
};

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
    const [isOpen, setOpen] = useState(false);

    return (
        <DialogTrigger isOpen={isOpen} onOpenChange={setOpen}>
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
                            setOpen(false);
                        }}
                    />
                </Suspense>
            )}
        </DialogTrigger>
    );
};
