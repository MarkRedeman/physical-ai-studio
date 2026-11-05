import { Button, Flex, ProgressBar, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { components, SchemaOptimizeModelJob } from '../../../api/openapi-spec';
import { JOBS_QUERY_KEY, useJobCache } from '../../jobs/use-job-cache';

const isActive = (job: SchemaOptimizeModelJob) => job.status === 'pending' || job.status === 'running';

export const OptimizeModel = ({
    modelId,
    backend,
    variant,
    operations,
}: {
    modelId: string;
    backend: components['schemas']['physicalai__export__backends__ExportBackend'];
    variant: string;
    operations: components['schemas']['VariantOperationDetail'][];
}) => {
    const { data: jobs = [], isPending, isError, refetch } = $api.useQuery('get', '/api/jobs');
    const { addJob } = useJobCache();
    const optimize = $api.useMutation('post', '/api/models/{model_id}/exports/{backend}/variants', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onSuccess: addJob,
    });
    const interrupt = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
    });

    const matchingJobs = jobs
        .filter(
            (job): job is SchemaOptimizeModelJob =>
                job.type === 'optimize_model' &&
                job.payload.model_id === modelId &&
                (job.payload.backend ?? 'openvino') === backend &&
                (job.payload.source_variant ?? 'original') === variant
        )
        .toSorted((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''));
    const activeJob = matchingJobs.findLast(isActive);
    const latestJob = matchingJobs.at(-1);

    return (
        <Flex direction='column' gap='size-100'>
            <Text>
                Create a standalone variant without changing the original. INT8 compresses weights, not activations. RTC
                requires a checkpoint; SnapFlow requires already-distilled weights. Validate policy accuracy before
                deployment.
            </Text>
            <Flex gap='size-100'>
                {operations.map((operation) => (
                    <div key={operation.operation} title={operation.reason ?? undefined}>
                        <Button
                            variant='secondary'
                            isDisabled={
                                !operation.available ||
                                isPending ||
                                isError ||
                                optimize.isPending ||
                                activeJob !== undefined
                            }
                            onPress={() => {
                                interrupt.reset();
                                optimize.mutate({
                                    params: { path: { model_id: modelId, backend } },
                                    body: { source_variant: variant, operation: operation.operation },
                                });
                            }}
                        >
                            {operation.operation === 'int8_sym'
                                ? 'Optimize to INT8'
                                : operation.operation === 'rtc'
                                  ? 'Export RTC'
                                  : 'Export SnapFlow'}
                        </Button>
                    </div>
                ))}
                {activeJob?.id && (
                    <Button
                        variant='secondary'
                        isDisabled={interrupt.isPending}
                        onPress={() => interrupt.mutate({ params: { path: { job_id: activeJob.id! } } })}
                    >
                        Cancel optimization
                    </Button>
                )}
            </Flex>
            {latestJob && (
                <div role='status'>
                    <Flex direction='column' gap='size-100'>
                        <Text>
                            Optimization {latestJob.status}: {latestJob.message}
                        </Text>
                        {isActive(latestJob) && (
                            <ProgressBar aria-label='Optimization progress' value={latestJob.progress} showValueLabel />
                        )}
                    </Flex>
                </div>
            )}
            {optimize.isError && (
                <div role='alert'>
                    <Text>
                        Could not start optimization. {getApiErrorMessage(optimize.error) ?? 'Please try again.'}
                    </Text>
                </div>
            )}
            {interrupt.isError && (
                <div role='alert'>
                    <Text>
                        Could not cancel optimization. {getApiErrorMessage(interrupt.error) ?? 'Please try again.'}
                    </Text>
                </div>
            )}
            {isError && (
                <Flex gap='size-100' alignItems='center'>
                    <div role='alert'>
                        <Text>Could not load optimization status.</Text>
                    </div>
                    <Button variant='secondary' onPress={() => refetch()}>
                        Retry status
                    </Button>
                </Flex>
            )}
        </Flex>
    );
};
