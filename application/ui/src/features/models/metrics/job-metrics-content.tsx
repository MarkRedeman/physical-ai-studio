import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { fetchSSE } from '../../../api/fetch-sse';

import { MetricsEntry } from './metrics-chart-utils';
import { MetricsGraphs } from './metrics-graphs.component';

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
        staleTime: Infinity,
    });

    return <MetricsGraphs data={query.data} />;
};
