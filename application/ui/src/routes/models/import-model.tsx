import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, FileTrigger, Flex, Heading, Text, TextField } from '@geti/ui';

import { $api } from '../../api/client';
import { SchemaModel } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';

export const ImportModelModal = (close: (model: SchemaModel | undefined) => void) => {
    const { id: project_id } = useProject();
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [name, setName] = useState<string>('');
    const [error, setError] = useState<string | null>(null);

    const importMutation = $api.useMutation('post', '/api/models:import', {
        onError: (err) => {
            setError(err instanceof Error ? err.message : 'Import failed');
        },
    });

    const onFileSelect = (files: FileList | null) => {
        if (!files || files.length === 0) return;
        const file = files[0];

        if (!file.name.endsWith('.zip')) {
            setError('Please select a .zip file');
            return;
        }

        setError(null);
        setSelectedFile(file);

        // Pre-fill model name from filename (strip .zip extension)
        if (!name) {
            setName(file.name.replace(/\.zip$/, ''));
        }
    };

    const submit = () => {
        if (!selectedFile || !name.trim()) return;

        setError(null);

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('project_id', project_id);
        formData.append('name', name.trim());

        importMutation.mutateAsync(
            {
                body: formData as never,
                bodySerializer: (body: unknown) => body as unknown as BodyInit,
            } as never,
        ).then((response) => {
            close(response as SchemaModel | undefined);
        });
    };

    return (
        <Dialog>
            <Heading>Import Model</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Flex direction='column' gap='size-100'>
                        <Text>Select a previously exported model archive (.zip):</Text>
                        <Flex alignItems='center' gap='size-150'>
                            <FileTrigger acceptedFileTypes={['.zip']} onSelect={onFileSelect}>
                                <Button variant='secondary'>Browse</Button>
                            </FileTrigger>
                            <Text UNSAFE_style={{ opacity: selectedFile ? 1 : 0.5 }}>
                                {selectedFile ? selectedFile.name : 'No file selected'}
                            </Text>
                        </Flex>
                    </Flex>
                    <TextField
                        label='Model name'
                        value={name}
                        onChange={setName}
                        isRequired
                    />
                    {error && (
                        <Text UNSAFE_style={{ color: 'var(--spectrum-negative-color-900)' }}>
                            {error}
                        </Text>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={submit}
                    isDisabled={!selectedFile || !name.trim() || importMutation.isPending}
                >
                    {importMutation.isPending ? 'Importing...' : 'Import'}
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
