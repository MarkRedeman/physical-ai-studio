import { useState } from 'react';

import {
    Button,
    Content,
    DialogContainer,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    IllustratedMessage,
    Text,
    View,
    Well,
} from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import useWebSocket from 'react-use-websocket';

import { $api, fetchClient } from '../../api/client';
import { SchemaJob, SchemaModel } from '../../api/openapi-spec';
import { LogsDialog } from '../../features/logs/show-logs.component';
import { useProjectId } from '../../features/projects/use-project';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { ImportModelModal } from './import-model';
import { ImportExportRow, TrainingHeader, TrainingRow } from './job-table.component';
import { ModelHeader, ModelRow } from './model-table.component';
import { SchemaTrainJob, TrainModelDialog } from './train-model-dialog';

const ModelList = ({
    models,
    jobs,
    onRetrain,
    onViewLogs,
    onExport,
}: {
    models: SchemaModel[];
    jobs: SchemaJob[];
    onRetrain: (model: SchemaModel) => void;
    onViewLogs: (model: SchemaModel) => void;
    onExport: (model: SchemaModel) => void;
}) => {
    const sortedModels = models.toSorted(
        (a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime()
    );

    const jobsById = new Map(jobs.map((j) => [j.id, j]));

    const deleteModelMutation = $api.useMutation('delete', '/api/models');

    const deleteModel = (model: SchemaModel) => {
        deleteModelMutation.mutate({ params: { query: { model_id: model.id! } } });
    };

    return (
        <View marginBottom={'size-600'}>
            <ModelHeader />
            {sortedModels.map((model) => (
                <ModelRow
                    key={model.id}
                    model={model}
                    trainingJob={model.train_job_id ? jobsById.get(model.train_job_id) : undefined}
                    onDelete={() => deleteModel(model)}
                    onRetrain={() => onRetrain(model)}
                    onViewLogs={() => onViewLogs(model)}
                    onExport={() => onExport(model)}
                />
            ))}
        </View>
    );
};

const JobList = ({ jobs, onDownload }: { jobs: SchemaJob[]; onDownload: (job: SchemaJob) => void }) => {
    const sortedJobs = jobs
        .filter((m) => m.status !== 'completed' || m.type === 'export')
        .toSorted((a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime());

    const interruptMutation = $api.useMutation('post', '/api/jobs/{job_id}:interrupt');
    const onInterrupt = (job: SchemaJob) => {
        if (job.id !== undefined) {
            interruptMutation.mutate({
                params: {
                    query: {
                        uuid: job.id,
                    },
                },
            });
        }
    };

    if (sortedJobs.length === 0) {
        return <></>;
    }

    const trainingJobs = sortedJobs.filter((j) => j.type === 'training') as SchemaTrainJob[];
    const importExportJobs = sortedJobs.filter((j) => j.type === 'import' || j.type === 'export');

    return (
        <View marginBottom={'size-600'}>
            <Heading level={4} marginBottom={'size-100'}>
                Jobs
            </Heading>

            {trainingJobs.length > 0 && (
                <>
                    <TrainingHeader />
                    {trainingJobs.map((job) => (
                        <TrainingRow key={job.id} trainJob={job} onInterrupt={() => onInterrupt(job)} />
                    ))}
                </>
            )}
            {importExportJobs.map((job) => (
                <ImportExportRow
                    key={job.id}
                    job={job}
                    onInterrupt={() => onInterrupt(job)}
                    onDownload={() => onDownload(job)}
                />
            ))}
        </View>
    );
};

const useProjectJobs = (project_id: string): SchemaJob[] => {
    const { data: allJobs } = $api.useQuery('get', '/api/jobs');

    return allJobs?.filter((j) => j.project_id === project_id) ?? [];
};

export const Index = () => {
    const { project_id } = useProjectId();
    const { data: models } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id } },
    });

    const jobs = useProjectJobs(project_id);
    const [retrainModel, setRetrainModel] = useState<SchemaModel | null>(null);
    const [logsSourceId, setLogsSourceId] = useState<string | undefined>();

    const handleViewLogs = (model: SchemaModel) => {
        if (model.train_job_id) {
            setLogsSourceId(`job-training-${model.train_job_id}`);
        }
    };

    const exportMutation = $api.useMutation('post', '/api/models/{model_id}:export');

    const handleExport = (model: SchemaModel) => {
        exportMutation.mutate(
            { params: { path: { model_id: model.id! } } },
            { onSuccess: (job) => addJob(job as SchemaJob) }
        );
    };

    const handleDownload = (job: SchemaJob) => {
        const payload = job.payload as { model_id?: string; model_name?: string };
        if (!payload.model_id) return;

        fetch(`/api/models/${payload.model_id}/export/download`).then(async (res) => {
            if (!res.ok) return;
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${payload.model_name ?? 'model'}.zip`;
            link.click();
            URL.revokeObjectURL(url);
        });
    };

    const {} = useWebSocket(fetchClient.PATH('/api/jobs/ws'), {
        shouldReconnect: () => true,
        onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
    });
    const client = useQueryClient();

    const updateJob = (job: SchemaJob) => {
        client.setQueryData<SchemaJob[]>(['get', '/api/jobs'], (old = []) => {
            return old.map((m) => (m.id === job.id ? job : m));
        });
    };

    const addJob = (job: SchemaJob) => {
        client.setQueryData<SchemaJob[]>(['get', '/api/jobs'], (old = []) => {
            return [...old, job];
        });
    };

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message_data = JSON.parse(data);
        if (message_data.event === 'JOB_UPDATE') {
            const message = message_data as { event: string; data: SchemaJob };
            if (message.data.project_id !== project_id) {
                return;
            }

            updateJob(message.data);
            if (message.data.status === 'completed') {
                client.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/models'] });
            }
        }
        if (message_data.event === 'MODEL_UPDATE') {
            client.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/models'] });
        }
    };

    const hasModels = models.length > 0;
    const hasJobs = jobs.length > 0;
    const showIllustratedMessage = !hasModels && !hasJobs;

    return (
        <Flex height='100%'>
            <Flex margin={'size-200'} direction={'column'} flex>
                <Heading level={4}>Models</Heading>
                <Divider size='S' marginTop='size-100' marginBottom={'size-100'} />
                {showIllustratedMessage ? (
                    <Well flex UNSAFE_style={{ backgroundColor: 'rgb(60,62,66)' }}>
                        <IllustratedMessage>
                            <EmptyIllustration />
                            <Content> Currently there are no trained models available. </Content>
                            <Text>If you&apos;ve recorded a dataset it&apos;s time to begin training your model. </Text>
                            <Heading>No trained models</Heading>
                            <View margin={'size-100'}>
                                <Flex gap='size-100'>
                                    <DialogTrigger>
                                        <Button variant='accent'>Train model</Button>
                                        {(close) => <TrainModelDialog close={close} />}
                                    </DialogTrigger>
                                    <DialogTrigger>
                                        <Button variant='secondary'>Import model</Button>
                                        {(close) =>
                                            ImportModelModal((job) => {
                                                if (job) addJob(job);
                                                close();
                                            })
                                        }
                                    </DialogTrigger>
                                </Flex>
                            </View>
                        </IllustratedMessage>
                    </Well>
                ) : (
                    <View margin={'size-300'}>
                        <Flex justifyContent={'end'} marginBottom='size-300' gap='size-100'>
                            <DialogTrigger>
                                <Button variant='secondary'>Import model</Button>
                                {(close) =>
                                    ImportModelModal((job) => {
                                        if (job) addJob(job);
                                        close();
                                    })
                                }
                            </DialogTrigger>
                            <DialogTrigger>
                                <Button variant='secondary'>Train model</Button>
                                {(close) => (
                                    <TrainModelDialog
                                        close={(job) => {
                                            if (job) addJob(job);
                                            close();
                                        }}
                                    />
                                )}
                            </DialogTrigger>
                        </Flex>
                        <JobList jobs={jobs} onDownload={handleDownload} />
                        {hasModels && (
                            <ModelList
                                models={models}
                                jobs={jobs}
                                onRetrain={setRetrainModel}
                                onViewLogs={handleViewLogs}
                                onExport={handleExport}
                            />
                        )}
                    </View>
                )}
            </Flex>
            <DialogContainer onDismiss={() => setRetrainModel(null)}>
                {retrainModel && (
                    <TrainModelDialog
                        baseModel={retrainModel}
                        close={(job) => {
                            if (job) addJob(job);
                            setRetrainModel(null);
                        }}
                    />
                )}
            </DialogContainer>
            <DialogContainer type='fullscreen' onDismiss={() => setLogsSourceId(undefined)}>
                {logsSourceId != null && (
                    <LogsDialog close={() => setLogsSourceId(undefined)} initialSourceId={logsSourceId} />
                )}
            </DialogContainer>
        </Flex>
    );
};
