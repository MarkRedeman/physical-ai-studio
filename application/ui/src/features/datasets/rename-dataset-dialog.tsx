import { useState, type FormEvent } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading, Item, Picker, Text, TextField } from '@geti-ui/ui';
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
    const [environmentToAddId, setEnvironmentToAddId] = useState<string>();
    const [cameraMappings, setCameraMappings] = useState<Record<string, string>>({});
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
    const { data: datasetCameraKeys = [] } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/recording-camera-keys',
        { params: { path: { dataset_id: dataset.id ?? '' } } },
        { enabled: dataset.id !== undefined }
    );
    const { data: environmentToAdd } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        { params: { path: { project_id: dataset.project_id, environment_id: environmentToAddId ?? '' } } },
        { enabled: environmentToAddId !== undefined }
    );
    const defaultAssociation = recordingEnvironments.find(({ is_default }) => is_default);
    const selectedDefaultId = defaultRecordingEnvironmentId ?? defaultAssociation?.id;
    const defaultMutation = $api.useMutation(
        'patch',
        '/api/dataset/{dataset_id}/recording-environments/{recording_environment_id}'
    );
    const addRecordingEnvironmentMutation = $api.useMutation(
        'post',
        '/api/dataset/{dataset_id}/recording-environments',
        {
            meta: {
                invalidates: [
                    [
                        'get',
                        '/api/dataset/{dataset_id}/recording-environments',
                        { params: { path: { dataset_id: dataset.id! } } },
                    ],
                ],
            },
        }
    );
    const availableEnvironments = environments.filter(
        ({ id }) => !recordingEnvironments.some(({ environment_id }) => environment_id === id)
    );
    const camerasToMap = environmentToAdd?.cameras ?? [];
    const requiresCameraMappings = datasetCameraKeys.length > 0;
    const hasCompleteCameraMappings =
        !requiresCameraMappings ||
        (camerasToMap.length > 0 &&
            camerasToMap.every((camera) => camera.id !== undefined && cameraMappings[camera.id] !== undefined) &&
            new Set(Object.values(cameraMappings)).size === camerasToMap.length);

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

    const addRecordingEnvironment = async () => {
        if (dataset.id === undefined || environmentToAddId === undefined) {
            return;
        }
        const recordingEnvironment = await addRecordingEnvironmentMutation.mutateAsync({
            params: { path: { dataset_id: dataset.id } },
            body: {
                environment_id: environmentToAddId,
                camera_mappings: camerasToMap.flatMap((camera) => {
                    if (camera.id === undefined) return [];
                    const datasetCameraKey = cameraMappings[camera.id];
                    return datasetCameraKey === undefined
                        ? []
                        : [{ environment_camera_id: camera.id, dataset_camera_key: datasetCameraKey }];
                }),
            },
        });
        setEnvironmentToAddId(undefined);
        setCameraMappings({});
        if (defaultAssociation === undefined) {
            setDefaultRecordingEnvironmentId(recordingEnvironment.id);
        }
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
                    <Picker
                        label='Add recording environment'
                        items={availableEnvironments}
                        selectedKey={environmentToAddId}
                        onSelectionChange={(id) => {
                            setEnvironmentToAddId(id?.toString());
                            setCameraMappings({});
                        }}
                    >
                        {(environment) => <Item key={environment.id}>{environment.name}</Item>}
                    </Picker>
                    {requiresCameraMappings && environmentToAddId !== undefined && (
                        <>
                            <Text>Map every environment camera to the camera key already used by this dataset.</Text>
                            {camerasToMap.map((camera) => {
                                if (camera.id === undefined) return null;
                                const selectedCameraKey = cameraMappings[camera.id];
                                const availableCameraKeys = datasetCameraKeys
                                    .filter(
                                        (cameraKey) =>
                                            cameraKey === selectedCameraKey ||
                                            !Object.entries(cameraMappings).some(
                                                ([cameraId, mappedKey]) =>
                                                    cameraId !== camera.id && mappedKey === cameraKey
                                            )
                                    )
                                    .map((cameraKey) => ({ id: cameraKey, name: cameraKey }));
                                return (
                                    <Picker
                                        key={camera.id}
                                        label={`Camera mapping for ${camera.name}`}
                                        items={availableCameraKeys}
                                        selectedKey={selectedCameraKey}
                                        onSelectionChange={(cameraKey) =>
                                            setCameraMappings((current) => {
                                                if (cameraKey === null) {
                                                    const next = { ...current };
                                                    delete next[camera.id!];
                                                    return next;
                                                }
                                                return { ...current, [camera.id!]: cameraKey.toString() };
                                            })
                                        }
                                    >
                                        {(cameraKey) => <Item key={cameraKey.id}>{cameraKey.name}</Item>}
                                    </Picker>
                                );
                            })}
                        </>
                    )}
                    <Button
                        type='button'
                        variant='secondary'
                        onPress={addRecordingEnvironment}
                        isDisabled={environmentToAddId === undefined || !hasCompleteCameraMappings}
                        isPending={addRecordingEnvironmentMutation.isPending}
                    >
                        Add environment
                    </Button>
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
