import { useMemo, useState } from 'react';

import {
    ActionButton,
    Button,
    DialogContainer,
    Flex,
    Grid,
    Icon,
    Item,
    Key,
    Menu,
    MenuTrigger,
    ProgressBar,
    Text,
    View,
} from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../api/client';
import { SchemaJob } from '../../api/openapi-spec';
import { LogsDialog } from '../../features/logs/logs-dialog';
import { useProjectId } from '../../features/projects/use-project';
import { SingleBadge, SplitBadge } from '../models/split-badge.component';
import { elapsedSince } from '../models/utils';
import { DatasetImportButton } from './dataset-import-button';

import classes from './dataset-import-jobs.module.scss';

const GRID_COLUMNS = ['2fr', '1fr', '1fr', 'auto'];

const asImportPayload = (payload: unknown): { step?: string } => {
    if (payload && typeof payload === 'object') {
        return payload as { step?: string };
    }
    return {};
};

const ImportJobHeader = () => {
    return (
        <Grid columns={GRID_COLUMNS} alignItems='center' width='100%' UNSAFE_className={classes.jobHeader}>
            <Text>Import job details</Text>
            <Text>Job ID</Text>
            <Text>Started</Text>
            <div />
        </Grid>
    );
};

const toStepLabel = (step?: string): string => {
    if (!step) {
        return 'unknown';
    }
    return step
        .split('_')
        .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
        .join(' ');
};

const ImportJobStatus = ({ job, step }: { job: SchemaJob; step: string }) => {
    const isRunning = job.status === 'running' || step === 'Importing Resource' || step === 'Ready To Commit';

    if (isRunning) {
        return (
            <View>
                <Flex gap='size-100' alignItems='center'>
                    <Text UNSAFE_style={{ fontWeight: 500 }}>Dataset import</Text>
                    <SplitBadge first='running' second={step} />
                </Flex>
                {job.created_at ? (
                    <Text UNSAFE_className={classes.jobInfo}>
                        Started: {new Date(job.created_at).toLocaleString()} | Elapsed: {elapsedSince(job.created_at)}
                    </Text>
                ) : null}
            </View>
        );
    }

    const color = job.status === 'failed' ? 'var(--spectrum-negative-visual-color)' : 'var(--energy-blue)';
    return (
        <View>
            <Flex gap='size-100' alignItems='center'>
                <Text UNSAFE_style={{ fontWeight: 500 }}>Dataset import</Text>
                <SingleBadge color={color} text={job.status} />
                <Text UNSAFE_className={classes.jobInfo}>• {step}</Text>
            </Flex>
        </View>
    );
};

const ImportJobMenu = ({ job, onViewLogs }: { job: SchemaJob; onViewLogs: () => void }) => {
    const { project_id } = useProjectId();
    const queryClient = useQueryClient();

    const cancelMutation = $api.useMutation('post', '/api/projects/{project_id}/imports/datasets/{job_id}:cancel', {
        onSuccess: async () => {
            await queryClient.invalidateQueries({ queryKey: ['get', '/api/jobs'] });
        },
    });
    const deleteMutation = $api.useMutation('delete', '/api/jobs/{job_id}', {
        onSuccess: async () => {
            await queryClient.invalidateQueries({ queryKey: ['get', '/api/jobs'] });
        },
    });

    const onAction = (key: Key) => {
        const action = key.toString();
        if (action === 'cancel') {
            cancelMutation.mutate({ params: { path: { project_id, job_id: job.id! } } });
        }
        if (action === 'delete') {
            deleteMutation.mutate({ params: { path: { job_id: job.id! } } });
        }
        if (action === 'logs') {
            onViewLogs();
        }
    };

    const payload = asImportPayload(job.payload);
    const isAwaitingUserInput = payload.step === 'waiting_for_user_input';
    const canCancel =
        isAwaitingUserInput || (job.status !== 'completed' && job.status !== 'failed' && job.status !== 'canceled');
    const canDelete = job.status === 'failed' || job.status === 'canceled';

    return (
        <MenuTrigger>
            <ActionButton
                isQuiet
                aria-label='Import job actions'
                isDisabled={cancelMutation.isPending || deleteMutation.isPending}
            >
                <Icon>
                    <MoreMenu />
                </Icon>
            </ActionButton>
            <Menu
                onAction={onAction}
                disabledKeys={[...(canCancel ? [] : ['cancel']), ...(canDelete ? [] : ['delete'])]}
            >
                <Item key='logs'>Logs</Item>
                <Item key='cancel'>Cancel</Item>
                <Item key='delete'>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};

const ImportJobRow = ({ job, onViewLogs }: { job: SchemaJob; onViewLogs: () => void }) => {
    const payload = asImportPayload(job.payload);
    const canResume = payload.step === 'waiting_for_user_input' || payload.step === 'uploaded';
    const step = toStepLabel(payload.step);
    const isRunning = job.status === 'running' || step === 'Importing Resource' || step === 'Ready To Commit';

    return (
        <View>
            <Grid columns={GRID_COLUMNS} alignItems='center' width='100%' UNSAFE_className={classes.jobRow}>
                <ImportJobStatus job={job} step={step} />
                <Text UNSAFE_className={classes.jobInfo}>{job.id}</Text>
                <Text UNSAFE_className={classes.jobInfo}>
                    {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
                </Text>
                <Flex gap='size-100' justifyContent='end'>
                    {canResume && <DatasetImportButton existingJobId={job.id!} buttonLabel='Resume' />}
                    <ImportJobMenu job={job} onViewLogs={onViewLogs} />
                </Flex>
            </Grid>

            {isRunning && (
                <ProgressBar
                    size='S'
                    UNSAFE_className={classes.progressBar}
                    width='100%'
                    value={job.progress ?? 10}
                    isIndeterminate={job.progress === undefined}
                />
            )}
        </View>
    );
};

export const DatasetImportJobs = () => {
    const { project_id } = useProjectId();
    const [listVisible, setListVisible] = useState(true);
    const [logsSourceId, setLogsSourceId] = useState<string | undefined>(undefined);
    const { data: jobs = [] } = $api.useQuery(
        'get',
        '/api/jobs',
        {
            params: undefined,
        },
        {
            refetchInterval: 2000,
        }
    );

    const importJobs = useMemo(() => {
        return jobs
            .filter((job) => job.project_id === project_id && job.type === 'dataset_import')
            .filter((job) => {
                const payload = asImportPayload(job.payload);
                if (payload.step === 'waiting_for_user_input') {
                    return true;
                }
                return job.status !== 'completed';
            })
            .sort((a, b) => new Date(b.created_at ?? 0).getTime() - new Date(a.created_at ?? 0).getTime());
    }, [jobs, project_id]);

    if (importJobs.length === 0) {
        return null;
    }

    return (
        <View marginBottom='size-200'>
            <Flex justifyContent='space-between' alignItems='center' marginBottom='size-100'>
                <Text UNSAFE_style={{ fontWeight: 600 }}>Pending dataset imports</Text>
                <Button variant='secondary' onPress={() => setListVisible(!listVisible)}>
                    {listVisible ? 'Hide' : 'Show'}
                </Button>
            </Flex>

            {listVisible && (
                <View marginTop='size-100'>
                    <ImportJobHeader />
                    {importJobs.map((job) => (
                        <ImportJobRow key={job.id} job={job} onViewLogs={() => setLogsSourceId(job.id)} />
                    ))}
                </View>
            )}

            <DialogContainer type='fullscreen' onDismiss={() => setLogsSourceId(undefined)}>
                {logsSourceId && (
                    <LogsDialog close={() => setLogsSourceId(undefined)} initialSourceId={`job-${logsSourceId}`} />
                )}
            </DialogContainer>
        </View>
    );
};
