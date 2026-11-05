// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { AlertDialog, Badge, Button, DialogTrigger, Flex, ProgressBar, Text } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { getApiErrorMessage } from '../../api/errors';
import { notify } from '../../components/notification/notification.component';
import { Table, TableColumn } from '../../components/table/table';
import { JobMenu, TrainingRow } from '../models/job-table/job-table';
import { JOBS_QUERY_KEY, SchemaJob } from './use-job-cache';

const COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: '2fr', header: 'Job' },
    { width: '1fr', header: 'Loss' },
    { width: '1fr', header: 'Policy / backend' },
    { width: '1fr', header: 'Dataset' },
    { width: '1fr', header: 'Environment' },
    { width: '1fr', header: 'Executor' },
    { width: '1fr' },
    { width: 'auto', align: 'end' },
];

type NonTrainingJob = Exclude<SchemaJob, { type: 'training' }>;

const GeneralJobRow = ({
    job,
    onInterrupt,
    onViewLogs,
}: {
    job: NonTrainingJob;
    onInterrupt: () => void;
    onViewLogs: () => void;
}) => {
    const modelId = job.type === 'optimize_model' ? job.payload.model_id : undefined;
    const { data } = $api.useQuery(
        'get',
        '/api/models/{model_id}',
        {
            params: { path: { model_id: modelId ?? '' } },
        },
        { enabled: modelId !== undefined }
    );
    const typeLabel = job.type === 'optimize_model' ? 'Model optimization' : 'Dataset import';
    const label =
        job.type === 'optimize_model'
            ? (data?.model.name ?? 'Model export')
            : (job.payload.dataset_name ?? job.payload.uploaded_archive_name ?? 'Import archive');
    const canStop = job.status === 'pending' || (job.type === 'optimize_model' && job.status === 'running');

    return (
        <Table.Row
            id={job.id}
            after={
                job.status === 'running' && (
                    <ProgressBar
                        aria-label={`${typeLabel} progress for ${label}`}
                        value={job.progress}
                        width='100%'
                        size='S'
                    />
                )
            }
        >
            <div />
            <Flex direction='column' gap='size-50'>
                <Text UNSAFE_style={{ fontWeight: 500 }}>{label}</Text>
                <Flex gap='size-100' wrap>
                    <Badge variant='neutral'>{typeLabel}</Badge>
                    <Badge variant={job.status === 'failed' ? 'negative' : 'neutral'}>{job.status}</Badge>
                </Flex>
                {job.type === 'optimize_model' && (
                    <Text>
                        {job.payload.source_variant} → {job.payload.target_variant}
                    </Text>
                )}
                {job.message && <Text>{job.message}</Text>}
            </Flex>
            <Text>—</Text>
            <Text>{job.type === 'optimize_model' ? job.payload.backend.toUpperCase() : job.payload.format_hint}</Text>
            <Text>{job.type === 'dataset_import' ? (job.payload.dataset_name ?? '—') : '—'}</Text>
            <Text>—</Text>
            <Text>Local</Text>
            <div>
                {canStop && (
                    <DialogTrigger>
                        <Button variant='secondary'>Stop</Button>
                        <AlertDialog
                            title={`Stop ${typeLabel.toLowerCase()}?`}
                            variant='destructive'
                            primaryActionLabel='Stop'
                            cancelLabel='Cancel'
                            onPrimaryAction={onInterrupt}
                        >
                            Unpublished output will be discarded. A running optimization may finish its current
                            operation before stopping.
                        </AlertDialog>
                    </DialogTrigger>
                )}
            </div>
            <JobMenu job={job} onViewLogs={onViewLogs} />
        </Table.Row>
    );
};

export const JobsTable = ({
    jobs,
    onViewLogs,
    emptyMessage,
}: {
    jobs: SchemaJob[];
    onViewLogs: (job: SchemaJob) => void;
    emptyMessage: ReactNode;
}) => {
    const interrupt = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onError: (error) => notify('error', getApiErrorMessage(error) ?? 'Could not stop the job.'),
    });
    const sorted = jobs.toSorted(
        (a, b) => (Date.parse(b.created_at ?? '') || 0) - (Date.parse(a.created_at ?? '') || 0)
    );
    if (sorted.length === 0) return <>{emptyMessage}</>;
    return (
        <Table columns={COLUMNS}>
            {sorted.map((job) =>
                job.type === 'training' ? (
                    <TrainingRow
                        key={job.id}
                        trainJob={job}
                        onInterrupt={() => interrupt.mutate({ params: { path: { job_id: job.id! } } })}
                        onViewLogs={() => onViewLogs(job)}
                    />
                ) : (
                    <GeneralJobRow
                        key={job.id}
                        job={job}
                        onInterrupt={() => interrupt.mutate({ params: { path: { job_id: job.id! } } })}
                        onViewLogs={() => onViewLogs(job)}
                    />
                )
            )}
        </Table>
    );
};
