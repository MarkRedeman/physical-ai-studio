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
import { DownloadProgressContent } from '../../../components/download-progress-content';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { isAbortError } from '../../utils/download';

const VALID_SOURCE_HINTS = ['auto', 'studio', 'lerobot_v2', 'lerobot_v3', 'trossen_sdk'] as const;
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

type ImportPayload = {
    step?: string;
    source_hint?: string;
    result_dataset_id?: string;
};

type ImportJobResponse = { id: string };

const asImportPayload = (payload: unknown): ImportPayload => {
    if (payload && typeof payload === 'object') {
        return payload as ImportPayload;
    }
    return {};
};

const submitDatasetImportTwoPhase = async ({
    projectId,
    file,
    source,
    abortRef,
    onUploadProgress,
}: {
    projectId: string;
    file: File;
    source: string;
    abortRef: React.MutableRefObject<XMLHttpRequest | null>;
    onUploadProgress: (progress: number | null) => void;
}): Promise<ImportJobResponse> => {
    const preparePath = fetchClient.PATH('/api/projects/{project_id}/imports/datasets:prepare', {
        params: { path: { project_id: projectId } },
    });

    return await new Promise<ImportJobResponse>((resolve, reject) => {
        const prepareXhr = new XMLHttpRequest();
        abortRef.current = prepareXhr;
        prepareXhr.open('POST', preparePath);
        prepareXhr.responseType = 'json';

        prepareXhr.onload = () => {
            if (prepareXhr.status < 200 || prepareXhr.status >= 300) {
                reject(new Error(`Failed to prepare import: ${prepareXhr.status}`));
                return;
            }

            const preparedJob = prepareXhr.response as ImportJobResponse;
            if (!preparedJob?.id) {
                reject(new Error('Failed to prepare import: missing job id'));
                return;
            }

            const uploadPath = fetchClient.PATH('/api/projects/{project_id}/imports/datasets/{job_id}:upload', {
                params: { path: { project_id: projectId, job_id: preparedJob.id } },
            });

            const uploadXhr = new XMLHttpRequest();
            abortRef.current = uploadXhr;
            uploadXhr.open('PUT', uploadPath);
            uploadXhr.responseType = 'json';

            uploadXhr.upload.onprogress = (event) => {
                if (event.lengthComputable && event.total > 0) {
                    onUploadProgress(Math.round((event.loaded / event.total) * 100));
                } else {
                    onUploadProgress(null);
                }
            };

            uploadXhr.onload = () => {
                if (uploadXhr.status >= 200 && uploadXhr.status < 300) {
                    const uploadedJob = uploadXhr.response as ImportJobResponse;
                    resolve({ id: uploadedJob?.id ?? preparedJob.id });
                } else {
                    reject(new Error(`Failed to upload dataset archive: ${uploadXhr.status}`));
                }
            };

            uploadXhr.onerror = () => reject(new Error('Failed to upload dataset archive'));
            uploadXhr.onabort = () => reject(new DOMException('Upload aborted', 'AbortError'));

            const uploadFormData = new FormData();
            uploadFormData.append('archive', file);
            uploadXhr.send(uploadFormData);
        };

        prepareXhr.onerror = () => reject(new Error('Failed to prepare dataset import job'));
        prepareXhr.onabort = () => reject(new DOMException('Upload aborted', 'AbortError'));

        const prepareFormData = new FormData();
        prepareFormData.append('source_hint', source);
        prepareXhr.send(prepareFormData);
    });
};

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
    const [uploadProgress, setUploadProgress] = useState<number | null>(null);
    const abortRef = useRef<XMLHttpRequest | null>(null);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
        };
    }, []);

    const uploadMutation = useMutation({
        mutationFn: async ({ file, source }: { file: File; source: string }) => {
            return await submitDatasetImportTwoPhase({
                projectId: project_id,
                file,
                source,
                abortRef,
                onUploadProgress: setUploadProgress,
            });
        },
        onMutate: () => {
            setUploadProgress(null);
        },
        onSettled: () => {
            abortRef.current = null;
        },
    });

    const startUpload = async (file: File) => {
        onFileSelected(file);

        try {
            const job = await uploadMutation.mutateAsync({ file, source: sourceHint });
            onUploaded(job.id);
        } catch (error) {
            if (isAbortError(error)) {
                return;
            }
        }
    };

    const handleFileList = (files: FileList | null) => {
        const file = files?.[0] ?? null;
        if (file !== null) {
            void startUpload(file);
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
                                <Button variant='secondary' isDisabled={uploadMutation.isPending}>
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
                    preparingMessage='Uploading dataset archive...'
                />
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onCancel}>
                    {uploadMutation.isPending ? 'Abort upload' : 'Cancel'}
                </Button>
            </ButtonGroup>
        </>
    );
};

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

    if (!hasTransitionedRef.current && job) {
        const payload = asImportPayload(job.payload);

        if (job.status === 'failed' || job.status === 'canceled') {
            hasTransitionedRef.current = true;

            if (payload.step === 'detecting_source') {
                onDetectionFailed(
                    job.message ?? 'Could not automatically detect the dataset format. Please select a format manually.'
                );
            } else {
                onFailed(job.message ?? 'Import failed during processing.');
            }
        } else if (job.status === 'completed' && payload.result_dataset_id) {
            hasTransitionedRef.current = true;
            onCompleted(payload.result_dataset_id);
        } else if (payload.step === 'waiting_for_user_input') {
            hasTransitionedRef.current = true;
            onReady();
        } else if (
            payload.step === 'ready_to_commit' ||
            payload.step === 'importing_resource' ||
            job.status === 'running'
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
    const [uploadProgress, setUploadProgress] = useState<number | null>(null);
    const abortRef = useRef<XMLHttpRequest | null>(null);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
        };
    }, []);

    const retryMutation = useMutation({
        mutationFn: async ({ file, source }: { file: File; source: string }) => {
            return await submitDatasetImportTwoPhase({
                projectId: project_id,
                file,
                source,
                abortRef,
                onUploadProgress: setUploadProgress,
            });
        },
        onMutate: () => {
            setUploadProgress(null);
        },
        onSettled: () => {
            abortRef.current = null;
        },
    });

    const onRetryUpload = async () => {
        if (!archive) {
            return;
        }

        try {
            const job = await retryMutation.mutateAsync({ file: archive, source: sourceHint });
            onRetry(job.id);
        } catch (error) {
            if (isAbortError(error)) {
                return;
            }
        }
    };

    const onCancel = () => {
        if (retryMutation.isPending) {
            abortRef.current?.abort();
            retryMutation.reset();
            setUploadProgress(null);
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
                    isError={retryMutation.isError && !isAbortError(retryMutation.error)}
                    isPending={retryMutation.isPending}
                    progress={uploadProgress}
                    errorMessage='Failed to upload dataset archive. Please try again.'
                    preparingMessage='Re-uploading dataset archive...'
                />
            </Content>

            <ButtonGroup>
                <Button variant='secondary' onPress={onCancel}>
                    {retryMutation.isPending ? 'Abort upload' : 'Cancel'}
                </Button>
                <Button
                    variant='accent'
                    onPress={onRetryUpload}
                    isPending={retryMutation.isPending}
                    isDisabled={archive === null}
                >
                    Retry with selected format
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
    const finalizeMutation = $api.useMutation('post', '/api/projects/{project_id}/imports/datasets/{job_id}:finalize');
    const [finalizeError, setFinalizeError] = useState<string | undefined>(undefined);
    const canFinalize = fields.environmentId !== undefined && fields.datasetName.trim().length > 0;

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
                <Text>Dataset analyzed. Review fields and click &quot;Finalize import&quot;.</Text>
                {finalizeError ? <Text>{finalizeError}</Text> : null}

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

    if (!hasCompletedRef.current && job) {
        const payload = asImportPayload(job.payload);

        if (job.status === 'completed' && payload.result_dataset_id) {
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
