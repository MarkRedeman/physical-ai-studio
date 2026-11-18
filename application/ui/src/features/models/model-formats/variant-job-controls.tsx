import { Button, Flex, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaModelVariantAction, SchemaOptimizeModelJob } from '../../../api/openapi-spec';
import { JOBS_QUERY_KEY, useJobCache } from '../../jobs/use-job-cache';

export const VariantJobControls = ({
    modelId,
    action,
    targetExists,
}: {
    modelId: string;
    action: SchemaModelVariantAction;
    targetExists: boolean;
}) => {
    const jobs = $api.useQuery('get', '/api/jobs');
    const { addJob } = useJobCache();
    const optimize = $api.useMutation('post', '/api/models/{model_id}/optimize', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onSuccess: addJob,
    });
    const exportModel = $api.useMutation('post', '/api/models/{model_id}/exports/{backend}', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onSuccess: addJob,
    });
    const interrupt = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
    });
    const matching = (jobs.data ?? [])
        .filter(
            (job): job is SchemaOptimizeModelJob =>
                job.type === 'optimize_model' &&
                job.payload.model_id === modelId &&
                (job.payload.operation ?? 'int8_sym') === action.operation &&
                (job.payload.backend ?? 'openvino') === action.backend
        )
        .toSorted((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
    const active = (jobs.data ?? []).find(
        (job) =>
            job.type === 'optimize_model' &&
            job.payload.model_id === modelId &&
            (job.status === 'pending' || job.status === 'running')
    );
    const isCompression = action.operation === 'int8_sym';
    const latest = active ?? matching[0];
    const error = optimize.error ?? exportModel.error ?? interrupt.error;

    return (
        <Flex direction='column' gap='size-100'>
            <Text>
                {isCompression
                    ? 'INT8 compresses weights, not activations. The original export stays unchanged.'
                    : 'Re-export the stored training checkpoint. This can take several minutes and additional memory.'}
            </Text>
            <Flex gap='size-100'>
                <Button
                    variant='secondary'
                    isPending={optimize.isPending || exportModel.isPending}
                    isDisabled={targetExists || jobs.isPending || jobs.isError || active !== undefined}
                    onPress={() => {
                        interrupt.reset();
                        if (isCompression) optimize.mutate({ params: { path: { model_id: modelId } } });
                        else exportModel.mutate({ params: { path: { model_id: modelId, backend: action.backend } } });
                    }}
                >
                    {isCompression ? 'Optimize to INT8' : 'Export model'}
                </Button>
                {active?.id && (
                    <Button
                        variant='secondary'
                        isPending={interrupt.isPending}
                        onPress={() => interrupt.mutate({ params: { path: { job_id: active.id! } } })}
                    >
                        Cancel job
                    </Button>
                )}
            </Flex>
            {latest && (
                <div role='status'>
                    Export {latest.status}: {latest.message}
                </div>
            )}
            {targetExists && <Text>This export is already available.</Text>}
            {error && <div role='alert'>{getApiErrorMessage(error) ?? 'Export failed. Please try again.'}</div>}
            {jobs.isError && (
                <Button variant='secondary' onPress={() => jobs.refetch()}>
                    Retry job status
                </Button>
            )}
        </Flex>
    );
};
