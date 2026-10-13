import { useState, type FormEvent } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading, Item, Picker, TextField } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';

import { $api } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';

type Dataset = SchemaDatasetOutput;

export const RenameDatasetDialog = ({
    dataset,
    onDone,
}: {
    dataset: Dataset;
    onDone: (dataset: SchemaDatasetOutput | undefined) => void;
}) => {
    const [name, setName] = useState(dataset.name);
    const [defaultTask, setDefaultTask] = useState(dataset.default_task);
    const [defaultRecordingEnvironmentId, setDefaultRecordingEnvironmentId] = useState<string>();
    const renameMutation = $api.useMutation('put', '/api/dataset/{dataset_id}', {
        meta: {
            invalidates: [
                ['get', '/api/dataset/{dataset_id}', { params: { path: { dataset_id: dataset.id! } } }],
                ['get', '/api/projects/{project_id}', { params: { path: { project_id: dataset.project_id } } }],
                ['get', '/api/projects'],
            ],
        },
    });
    const { data: recordingEnvironments = [] } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/recording-environments',
        { params: { path: { dataset_id: dataset.id ?? '' } } },
        { enabled: dataset.id !== undefined }
    );
    const { data: environments = [] } = $api.useQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id: dataset.project_id } },
    });
    const defaultAssociation = recordingEnvironments.find(({ is_default }) => is_default);
    const selectedDefaultId = defaultRecordingEnvironmentId ?? defaultAssociation?.id;
    const defaultMutation = $api.useMutation(
        'patch',
        '/api/dataset/{dataset_id}/recording-environments/{recording_environment_id}'
    );

    const isSaveDisabled =
        isEmpty(name.trim()) ||
        (name === dataset.name && defaultTask === dataset.default_task && selectedDefaultId === defaultAssociation?.id);

    const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const trimmedName = name.trim();
        if (trimmedName.length === 0 || dataset.id === undefined) {
            return;
        }

        const updated = await renameMutation.mutateAsync({
            params: {
                path: {
                    dataset_id: dataset.id,
                },
            },
            body: {
                name: trimmedName,
                default_task: defaultTask,
            },
        });

        if (selectedDefaultId !== undefined && selectedDefaultId !== defaultAssociation?.id) {
            await defaultMutation.mutateAsync({
                params: { path: { dataset_id: dataset.id, recording_environment_id: selectedDefaultId } },
                body: { is_default: true },
            });
        }

        onDone(updated);
    };

    return (
        <form onSubmit={onSubmit}>
            <Dialog width='size-6000'>
                <Heading>Edit dataset</Heading>
                <Divider />
                <Content>
                    <TextField
                        // eslint-disable-next-line jsx-a11y/no-autofocus
                        autoFocus
                        isRequired
                        width='100%'
                        label='Dataset name'
                        value={name}
                        onChange={setName}
                    />
                    <TextField label='Default task' value={defaultTask} onChange={setDefaultTask} />
                    <Picker
                        label='Default recording environment'
                        items={recordingEnvironments}
                        selectedKey={selectedDefaultId}
                        onSelectionChange={(id) => setDefaultRecordingEnvironmentId(id?.toString())}
                    >
                        {(association) => {
                            const environment = environments.find(({ id }) => id === association.environment_id);
                            return <Item key={association.id}>{environment?.name ?? 'Unavailable environment'}</Item>;
                        }}
                    </Picker>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={() => onDone(undefined)}>
                        Cancel
                    </Button>
                    <Button
                        variant='accent'
                        type='submit'
                        isDisabled={isSaveDisabled}
                        isPending={renameMutation.isPending}
                    >
                        Save
                    </Button>
                </ButtonGroup>
            </Dialog>
        </form>
    );
};
