import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    DropZone,
    FileTrigger,
    Flex,
    Heading,
    InlineAlert,
    ProgressCircle,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';

import { SchemaDatasetImportJob } from '../../../api/openapi-spec';
import { isAbortError } from '../../utils/download';
import { useDatasetUpload } from './use-dataset-upload';

interface ImportStepEditingProps {
    importJob: SchemaDatasetImportJob | undefined;
    project_id: string;
    onClose: () => void;
    datasetName: string;
    onDatasetNameChange: (value: string) => void;
    onUploaded: (jobId: string) => void;
}

export const ImportStepEditing = ({
    importJob,
    project_id,
    onClose,
    datasetName,
    onDatasetNameChange,
    onUploaded,
}: ImportStepEditingProps) => {
    const [archive, setArchive] = useState<File | null>(null);

    const errorMessage =
        importJob?.status === 'failed'
            ? (importJob.message ?? 'Import failed during processing.')
            : importJob?.status === 'canceled'
              ? 'Import was canceled.'
              : undefined;

    const { upload, abort, progress, mutation } = useDatasetUpload(project_id);
    const canUpload = archive !== null && datasetName.trim().length > 0;

    const startUpload = async (file: File) => {
        const jobId = await upload(file, 'auto', datasetName.trim());
        if (jobId) {
            onUploaded(jobId);
        }
    };

    const onUpload = async () => {
        if (archive === null || !canUpload) {
            return;
        }
        await startUpload(archive);
    };

    const onFileSelected = (file: File) => {
        setArchive(file);
        if (datasetName.trim().length === 0) {
            const suggestion = file.name.endsWith('.zip') ? file.name.slice(0, -4) : file.name;
            onDatasetNameChange(suggestion);
        }
    };

    const handleFileList = (files: FileList | null) => {
        const file = files?.[0] ?? null;
        if (file === null) {
            return;
        }

        onFileSelected(file);
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
                {errorMessage ? (
                    <InlineAlert variant='negative'>
                        <Heading>Import setup error</Heading>
                        <Content>{errorMessage}</Content>
                    </InlineAlert>
                ) : null}

                <Flex direction='column' gap='size-200'>
                    <TextField
                        isRequired
                        width='100%'
                        label='Dataset name'
                        value={datasetName}
                        onChange={onDatasetNameChange}
                        isDisabled={mutation.isPending}
                    />

                    <DropZone
                        isFilled={archive !== null}
                        onDrop={async (e) => {
                            if (mutation.isPending) {
                                return;
                            }
                            const fileItem = e.items.find((item) => item.kind === 'file');
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
                                        aria-label='Uploading dataset archive'
                                    />
                                )}
                                <Text>{progress === null ? 'Uploading dataset archive...' : `${progress}%`}</Text>
                            </Flex>
                        ) : (
                            <Flex direction='column' gap='size-100'>
                                {archive !== null ? (
                                    <Text>{archive.name}</Text>
                                ) : (
                                    <Text>Drop a .zip archive here or click to browse</Text>
                                )}
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
                </Flex>

                {mutation.isError && !isAbortError(mutation.error) ? (
                    <InlineAlert variant='negative'>
                        <Heading>Import error</Heading>
                        <Content>Failed to upload dataset archive. Please try again.</Content>
                    </InlineAlert>
                ) : null}
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onCancel}>
                    {mutation.isPending ? 'Abort upload' : 'Cancel'}
                </Button>
                <Button
                    variant='accent'
                    onPress={onUpload}
                    isPending={mutation.isPending}
                    isDisabled={!canUpload || mutation.isPending}
                >
                    Upload
                </Button>
            </ButtonGroup>
        </>
    );
};
