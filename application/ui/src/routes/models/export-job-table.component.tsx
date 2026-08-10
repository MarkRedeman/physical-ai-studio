import { ActionButton, Flex, Grid, Item, Key, Menu, MenuTrigger, ProgressBar, Text, View } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { $api } from '../../api/client';
import type { SchemaModelExportJob } from '../../api/openapi-spec';
import { ElapsedDuration } from '../../components/elapsed-duration.component';
import { CollapsableRow } from './collapsable-row.component';
import { SingleBadge } from './split-badge.component';
import { durationBetween } from './utils';

import classes from './model-table.module.css';

const EXPORT_GRID_COLUMNS = '2fr 1fr auto';

export const ExportHeader = () => (
    <Grid columns={EXPORT_GRID_COLUMNS} alignItems='center' width='100%' UNSAFE_className={classes.modelHeader}>
        <Text>Model</Text>
        <Text>Formats</Text>
        <div />
    </Grid>
);

const ExportJobMenu = ({ job, onViewLogs }: { job: SchemaModelExportJob; onViewLogs: () => void }) => {
    const deleteJobMutation = $api.useMutation('delete', '/api/jobs/{job_id}', {
        meta: {
            invalidates: [['get', '/api/jobs']],
        },
    });

    const onAction = (key: Key) => {
        if (key === 'logs') {
            onViewLogs();
        }
        if (key === 'delete' && job.id !== undefined) {
            deleteJobMutation.mutate({ params: { path: { job_id: job.id } } });
        }
    };

    return (
        <MenuTrigger>
            <ActionButton
                isQuiet
                UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}
                aria-label='Export job options'
                isDisabled={deleteJobMutation.isPending}
            >
                <MoreMenu />
            </ActionButton>
            <Menu
                onAction={onAction}
                disabledKeys={job.status === 'running' || job.status === 'pending' ? ['delete'] : []}
            >
                <Item key='logs'>Logs</Item>
                <Item key='delete'>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};

export const ExportRow = ({
    job,
    modelName,
    onViewLogs,
}: {
    job: SchemaModelExportJob;
    modelName: string;
    onViewLogs: () => void;
}) => {
    const active = job.status === 'pending' || job.status === 'running';
    const formats = job.payload.backends.map((backend) => backend.toUpperCase()).join(', ');
    const color = job.status === 'failed' ? 'var(--spectrum-negative-visual-color)' : 'var(--energy-blue)';
    const statusDetail = [
        `Status: ${job.status}`,
        `Formats: ${formats}`,
        job.message,
        job.extra_info ? JSON.stringify(job.extra_info, null, 2) : null,
    ]
        .filter((detail): detail is string => detail !== null && detail !== '')
        .join('\n\n');

    return (
        <View>
            <CollapsableRow
                header={
                    <Grid
                        columns={EXPORT_GRID_COLUMNS}
                        alignItems='center'
                        width='100%'
                        UNSAFE_className={classes.modelRow}
                    >
                        <View>
                            <Flex alignItems='center' gap='size-100' wrap>
                                <Text UNSAFE_style={{ fontWeight: 500 }}>{modelName}</Text>
                                <SingleBadge color={color} text={job.status} />
                            </Flex>
                            {job.start_time ? (
                                <Text UNSAFE_className={classes.modelInfo}>
                                    {active ? (
                                        <>
                                            Started: {new Date(job.start_time).toLocaleString()} | Elapsed:{' '}
                                            <ElapsedDuration date={job.start_time} />
                                        </>
                                    ) : job.end_time ? (
                                        <>Elapsed: {durationBetween(job.start_time, job.end_time)}</>
                                    ) : null}
                                </Text>
                            ) : null}
                        </View>
                        <Text>{formats}</Text>
                        <View justifySelf='end'>
                            <ExportJobMenu job={job} onViewLogs={onViewLogs} />
                        </View>
                    </Grid>
                }
            >
                <View backgroundColor='gray-100' padding='size-150'>
                    <pre style={{ margin: 0, overflowWrap: 'anywhere', whiteSpace: 'pre-wrap' }}>{statusDetail}</pre>
                </View>
            </CollapsableRow>
            {active && (
                <ProgressBar size='S' UNSAFE_className={classes.progressBar} value={job.progress} width='100%' />
            )}
        </View>
    );
};
