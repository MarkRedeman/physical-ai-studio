import { useState } from 'react';

import { Flex, ProgressCircle, Text, View } from '@geti-ui/ui';
import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../../api/client';
import { SchemaDatasetImportJob, SchemaTrainJob } from '../../../api/openapi-spec';

const DatasetImportJobStatus = ({ job }: { job: SchemaDatasetImportJob }) => {
    if (job?.status !== 'pending' && job?.status !== 'running') {
        return null;
    }

    return (
        <Flex gap='size-100'>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-700)',
                }}
            >
                Status:
            </Text>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-800)',
                }}
            >
                Importing dataset
            </Text>
        </Flex>
    );
};

const TrainingJobStatus = ({ job }: { job: SchemaTrainJob }) => {
    if (job?.status !== 'pending' && job?.status !== 'running') {
        return null;
    }

    return (
        <View width='100%' height='100%'>
            <Flex gap='size-100' alignItems={'center'} height='100%'>
                <ProgressCircle value={job.progress} size='S' />
                <Text
                    UNSAFE_style={{
                        color: 'var(--spectrum-global-color-gray-700)',
                    }}
                >
                    Status:
                </Text>
                <Text
                    UNSAFE_style={{
                        color: 'var(--spectrum-global-color-gray-800)',
                    }}
                >
                    Training {job.payload.model_name}
                </Text>
                <Text
                    UNSAFE_style={{
                        color: 'var(--spectrum-global-color-gray-700)',
                    }}
                >
                    {job.progress}%
                </Text>
            </Flex>
        </View>
    );
};

export const JobStatus = () => {
    const [job, setJob] = useState<null | SchemaTrainJob | SchemaDatasetImportJob>(null);

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message_data = JSON.parse(data);

        if (message_data.event === 'JOB_UPDATE') {
            setJob(message_data.data as SchemaTrainJob | SchemaDatasetImportJob);
        }
    };

    const {} = useWebSocket(fetchClient.PATH('/api/jobs/ws'), {
        shouldReconnect: () => true,
        onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
    });

    if (job?.type === 'training') {
        return <TrainingJobStatus job={job} />;
    }

    if (job?.type === 'dataset_import') {
        return <DatasetImportJobStatus job={job} />;
    }

    return null;
};
