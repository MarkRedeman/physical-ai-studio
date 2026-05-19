import { useEffect, useRef, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    DropZone,
    FileTrigger,
    Flex,
    Heading,
    InlineAlert,
    Item,
    Key,
    Picker,
    ProgressCircle,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';
import { useMutation, useQueryClient } from '@tanstack/react-query';

import { $api, fetchClient } from '../../api/client';
import { notify } from '../../components/notification/notification.component';
import { useProjectId } from '../../features/projects/use-project';
import { isAbortError } from '../utils/download';

interface UploadVariables {
    file: File;
    modelName: string;
    datasetId: string;
}

interface UseModelUploadResult {
    upload: (variables: UploadVariables) => Promise<void>;
    abort: () => void;
    progress: number | null;
    mutation: ReturnType<typeof useMutation<void, Error, UploadVariables>>;
}

const getImportErrorMessage = (response: unknown, status: number): string => {
    if (typeof response === 'object' && response !== null) {
        const maybeMessage = (response as { message?: unknown }).message;
        if (typeof maybeMessage === 'string' && maybeMessage.trim().length > 0) {
            return maybeMessage;
        }

        const detail = (response as { detail?: unknown }).detail;
        if (typeof detail === 'string' && detail.trim().length > 0) {
            return detail;
        }
    }

    return `Failed to import model archive: ${status}`;
};

const getErrorMessage = (error: unknown, fallbackMessage: string): string => {
    if (error instanceof Error && error.message.trim().length > 0) {
        return error.message;
    }
    return fallbackMessage;
};

const useModelUpload = (projectId: string): UseModelUploadResult => {
    const [progress, setProgress] = useState<number | null>(null);
    const abortRef = useRef<XMLHttpRequest | null>(null);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
        };
    }, []);

    const mutation = useMutation({
        mutationFn: async ({ file, modelName, datasetId }: UploadVariables) => {
            const uploadPath = fetchClient.PATH('/api/models:import');

            await new Promise<void>((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                abortRef.current = xhr;

                xhr.open('POST', uploadPath);
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
                        resolve();
                    } else {
                        reject(new Error(getImportErrorMessage(xhr.response, xhr.status)));
                    }
                };

                xhr.onerror = () => reject(new Error('Failed to import model archive'));
                xhr.onabort = () => reject(new DOMException('Upload aborted', 'AbortError'));

                const fd = new FormData();
                fd.append('archive', file);
                fd.append('project_id', projectId);
                fd.append('model_name', modelName);
                fd.append('dataset_id', datasetId);
                fd.append('version', '1');

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

    const upload = async (variables: UploadVariables): Promise<void> => {
        await mutation.mutateAsync(variables);
    };

    const abort = () => {
        abortRef.current?.abort();
        mutation.reset();
        setProgress(null);
    };

    return { upload, abort, progress, mutation };
};

interface ImportModelDialogProps {
    onClose: () => void;
    onImportCompleted?: () => void;
}

export const ImportModelDialog = ({ onClose, onImportCompleted }: ImportModelDialogProps) => {
    const queryClient = useQueryClient();
    const { project_id } = useProjectId();
    const { data: project } = $api.useQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id } },
    });

    const datasets = project?.datasets ?? [];

    const [modelName, setModelName] = useState('');
    const [selectedDataset, setSelectedDataset] = useState<Key | null>(null);
    const [archive, setArchive] = useState<File | null>(null);

    const { upload, abort, progress, mutation } = useModelUpload(project_id);

    const onFileSelected = (file: File) => {
        setArchive(file);

        if (modelName.trim().length === 0) {
            const suggestedName = file.name.endsWith('.zip') ? file.name.slice(0, -4) : file.name;
            setModelName(suggestedName);
        }
    };

    const handleFileList = (files: FileList | null) => {
        const file = files?.[0] ?? null;

        if (file === null) {
            return;
        }

        onFileSelected(file);
    };

    const canImport =
        archive !== null &&
        modelName.trim().length > 0 &&
        selectedDataset !== null &&
        !mutation.isPending;

    const onImport = async () => {
        if (!canImport || archive === null || selectedDataset === null) {
            return;
        }

        try {
            await upload({
                file: archive,
                modelName: modelName.trim(),
                datasetId: selectedDataset.toString(),
            });

            notify('success', 'Model import started successfully');
            await queryClient.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/models'] });
            onImportCompleted?.();
            onClose();
        } catch (error) {
            if (!isAbortError(error)) {
                notify('error', getErrorMessage(error, 'Failed to import model archive'));
            }
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
        <Dialog size='L'>
            <Heading>Import model</Heading>
            <Divider />

            <Content>
                <Flex direction='column' gap='size-200'>
                    <TextField
                        isRequired
                        width='100%'
                        label='Model name'
                        value={modelName}
                        onChange={setModelName}
                        isDisabled={mutation.isPending}
                    />

                    <Picker
                        isRequired
                        width='100%'
                        label='Dataset'
                        selectedKey={selectedDataset}
                        onSelectionChange={setSelectedDataset}
                        isDisabled={mutation.isPending}
                    >
                        {datasets.map((dataset) => (
                            <Item key={dataset.id}>{dataset.name}</Item>
                        ))}
                    </Picker>

                    <DropZone
                        isFilled={archive !== null}
                        onDrop={async (event) => {
                            if (mutation.isPending) {
                                return;
                            }

                            const fileItem = event.items.find((item) => item.kind === 'file');

                            if (fileItem?.kind !== 'file') {
                                return;
                            }

                            const file = await fileItem.getFile();
                            if (file.name.endsWith('.zip')) {
                                onFileSelected(file);
                            }
                        }}
                    >
                        {mutation.isPending ? (
                            <Flex direction='column' alignItems='center' justifyContent='center' gap='size-100'>
                                {progress !== null && (
                                    <ProgressCircle
                                        value={progress}
                                        minValue={0}
                                        maxValue={100}
                                        size='M'
                                        aria-label='Uploading model archive'
                                    />
                                )}
                                <Text>{progress === null ? 'Uploading model archive...' : `${progress}%`}</Text>
                            </Flex>
                        ) : (
                            <Flex direction='column' gap='size-100'>
                                {archive !== null ? <Text>{archive.name}</Text> : <Text>Drop a .zip archive here</Text>}
                                <View>
                                    <FileTrigger acceptedFileTypes={['.zip']} onSelect={handleFileList}>
                                        <Button variant='secondary'>
                                            {archive !== null ? 'Choose a different file' : 'Browse'}
                                        </Button>
                                    </FileTrigger>
                                </View>
                            </Flex>
                        )}
                    </DropZone>

                    {mutation.isError && !isAbortError(mutation.error) ? (
                        <InlineAlert variant='negative'>
                            <Heading>Import failed</Heading>
                            <Content>{getErrorMessage(mutation.error, 'Failed to upload model archive.')}</Content>
                        </InlineAlert>
                    ) : null}
                </Flex>
            </Content>

            <ButtonGroup>
                <Button variant='secondary' onPress={onCancel}>
                    {mutation.isPending ? 'Abort upload' : 'Cancel'}
                </Button>
                <Button variant='accent' onPress={onImport} isDisabled={!canImport} isPending={mutation.isPending}>
                    Import
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

export type { ImportModelDialogProps };
