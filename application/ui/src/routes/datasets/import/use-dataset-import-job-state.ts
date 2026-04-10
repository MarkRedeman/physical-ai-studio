import { skipToken } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import type { SchemaDatasetImportJob, SchemaDatasetImportJobPayload } from '../../../api/openapi-spec';

export const VALID_FORMAT_HINTS = ['auto', 'lerobot_v2', 'lerobot_v3'] as const;
export type FormatHint = (typeof VALID_FORMAT_HINTS)[number];

export interface FinalizeFields {
    defaultTask: string;
    environmentId: string | undefined;
}

export interface DraftManifestSummary {
    source_type?: string;
    source_format_version?: string;
    statistics?: {
        episode_count?: number;
        frame_count?: number;
    };
    dataset_schema?: {
        cameras?: Array<{
            name?: string;
            width?: number;
            height?: number;
            fps?: number;
        }>;
        robots?: Array<{
            name?: string;
            type?: string;
            joints?: string[];
        }>;
    };
    warnings?: string[];
    missing_fields?: string[];
}

export type DatasetImportJobSnapshot = {
    status: SchemaDatasetImportJob['status'];
    message?: string | null;
    type: string;
    payload: SchemaDatasetImportJobPayload;
};

export const useDatasetImportJobQuery = (importJobId: string | undefined) => {
    return $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        {
            params: { path: { job_id: importJobId ?? '' } },
        },
        {
            enabled: importJobId !== undefined,
            refetchInterval: 1000,
            // TODO: check if this is allowed
            //select: (job): job is SchemaDatasetImportJob => {
            select: (job) => {
                return job.type === 'dataset_import' ? job : skipToken;
            },
        }
    );
};
