import { useMemo } from 'react';

import { Flex } from '@geti-ui/ui';
import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import { fetchSSE } from '../../api/fetch-sse';
import { MetricGraph } from './metrics-graph.component';

export interface MetricsEntry {
    epoch: number;
    step: number;
    train_loss: number | null | undefined;
    train_loss_step: number | null | undefined;
    val_loss: number | null | undefined;
    'lr-AdamW': number | null | undefined;
}

export type MetricPoint = { x: number; y: number };

export const filterLossStepMetrics = (data?: MetricsEntry[]) => {
    if (!data) return [];
    return data.flatMap((entry): MetricPoint[] => {
        // Prefer the per-step train/loss. Fall back to train/loss_step, which ACT
        // logged per-step historically, so jobs still streaming from older runs
        // keep charting.
        const y = entry.train_loss ?? entry.train_loss_step;
        return y == null ? [] : [{ x: entry.step, y }];
    });
};

export const filterValLossMetrics = (data?: MetricsEntry[]) => {
    if (!data) return [];
    return data.flatMap((entry): MetricPoint[] =>
        entry.val_loss == null ? [] : [{ x: entry.step, y: entry.val_loss }]
    );
};

export const filterLrMetrics = (data?: MetricsEntry[]) => {
    if (!data) return [];
    return data.flatMap((entry): MetricPoint[] =>
        entry['lr-AdamW'] == null ? [] : [{ x: entry.step, y: entry['lr-AdamW'] }]
    );
};

const formatLearningRateTick = (value: number) => {
    if (value === 0) return '0';
    return value.toExponential(1);
};

// The metrics stream is long-lived. ``staleTime: Infinity`` keeps it from being
// auto-refetched as stale data, but the stream must still be re-established on
// remount (tab switches) and after a dropped connection.
const METRICS_QUERY_OPTIONS = {
    staleTime: Infinity,
    refetchOnMount: 'always' as const,
    retry: true,
};

const MetricsGraphs = ({ data }: { data?: MetricsEntry[] }) => {
    const lossStepMetrics = useMemo(() => filterLossStepMetrics(data), [data]);
    const valLossMetrics = useMemo(() => filterValLossMetrics(data), [data]);
    const lrMetrics = useMemo(() => filterLrMetrics(data), [data]);

    return (
        <Flex wrap gap='size-200'>
            <MetricGraph title={'Loss'} yAxisLabel={'Loss'} xAxisLabel='Step' data={lossStepMetrics} />
            <MetricGraph
                title={'Validation Loss'}
                yAxisLabel={'Validation Loss'}
                xAxisLabel='Step'
                data={valLossMetrics}
            />
            {lrMetrics.length > 0 && (
                <MetricGraph
                    title={'Learning Rate'}
                    yAxisLabel={'Learning Rate'}
                    xAxisLabel='Step'
                    data={lrMetrics}
                    yTickFormatter={formatLearningRateTick}
                />
            )}
        </Flex>
    );
};

export const JobMetricsContent = ({ jobId }: { jobId: string }) => {
    const query = useQuery({
        queryKey: ['get', '/api/models/{job_id}/model_metrics', jobId],
        queryFn: streamedQuery({
            streamFn: (context) => {
                const url = fetchClient.PATH('/api/jobs/{job_id}/model_metrics', {
                    params: { path: { job_id: jobId } },
                });

                return fetchSSE<MetricsEntry>(url, { signal: context.signal });
            },
        }),
        ...METRICS_QUERY_OPTIONS,
    });

    return <MetricsGraphs data={query.data} />;
};

export const MetricsContent = ({ modelId }: { modelId: string }) => {
    const query = useQuery({
        queryKey: ['get', '/api/models/{model_id}/metrics', modelId],
        queryFn: streamedQuery({
            streamFn: (context) => {
                const url = fetchClient.PATH('/api/models/{model_id}/metrics', {
                    params: { path: { model_id: modelId } },
                });

                return fetchSSE<MetricsEntry>(url, { signal: context.signal });
            },
        }),
        ...METRICS_QUERY_OPTIONS,
    });

    return <MetricsGraphs data={query.data} />;
};
