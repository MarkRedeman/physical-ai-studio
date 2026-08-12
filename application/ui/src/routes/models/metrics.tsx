import { useMemo } from 'react';

import { Flex } from '@geti-ui/ui';
import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import { fetchSSE } from '../../api/fetch-sse';
import { MetricGraph, type MetricGraphPoint, type MetricSeries } from './metrics-graph.component';

export interface MetricsEntry {
    epoch: number;
    step: number;
    train_loss: number | null | undefined;
    train_loss_step: number | null | undefined;
    val_loss: number | null | undefined;
    'lr-AdamW': number | null | undefined;
}

/** Merge per-step train and validation loss into rows the combined chart plots. */
export const buildLossMetrics = (data?: MetricsEntry[]): MetricGraphPoint[] => {
    if (!data) return [];
    const rows = new Map<number, MetricGraphPoint>();
    for (const entry of data) {
        // Prefer the per-step train/loss. Fall back to train/loss_step, which ACT
        // logged per-step historically, so jobs still streaming from older runs
        // keep charting.
        const train = entry.train_loss ?? entry.train_loss_step;
        if (train == null && entry.val_loss == null) {
            continue;
        }
        const row = rows.get(entry.step) ?? { x: entry.step };
        if (train != null) {
            row.train = train;
        }
        if (entry.val_loss != null) {
            row.val = entry.val_loss;
        }
        rows.set(entry.step, row);
    }
    return [...rows.values()].sort((a, b) => a.x - b.x);
};

export const filterLrMetrics = (data?: MetricsEntry[]): MetricGraphPoint[] => {
    if (!data) return [];
    return data.flatMap((entry): MetricGraphPoint[] => {
        const lr = entry['lr-AdamW'];
        return lr == null ? [] : [{ x: entry.step, lr }];
    });
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

const TRAIN_COLOR = 'var(--energy-blue)';
const VALIDATION_COLOR = 'var(--spectrum-semantic-negative-color-default, #e34850)';

const LOSS_SERIES: MetricSeries[] = [
    { dataKey: 'train', name: 'Train Loss', color: TRAIN_COLOR },
    { dataKey: 'val', name: 'Validation Loss', color: VALIDATION_COLOR },
];

const LR_SERIES: MetricSeries[] = [{ dataKey: 'lr', name: 'Learning Rate', color: TRAIN_COLOR }];

const MetricsGraphs = ({ data }: { data?: MetricsEntry[] }) => {
    const lossMetrics = useMemo(() => buildLossMetrics(data), [data]);
    const lrMetrics = useMemo(() => filterLrMetrics(data), [data]);

    return (
        <Flex wrap gap='size-200'>
            <MetricGraph title={'Loss'} yAxisLabel={'Loss'} xAxisLabel='Step' data={lossMetrics} series={LOSS_SERIES} />
            {lrMetrics.length > 0 && (
                <MetricGraph
                    title={'Learning Rate'}
                    yAxisLabel={'Learning Rate'}
                    xAxisLabel='Step'
                    data={lrMetrics}
                    series={LR_SERIES}
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
