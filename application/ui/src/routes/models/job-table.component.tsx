import { Button, Flex, Grid, ProgressBar, Text, View } from '@geti/ui';

import { SchemaJob } from '../../api/openapi-spec';
import { GRID_COLUMNS } from './constants';
import { SingleBadge, SplitBadge } from './split-badge.component';
import { SchemaTrainJob } from './train-model-dialog';
import { durationBetween, elapsedSince } from './utils';

import classes from './model-table.module.scss';

export const TrainingHeader = () => {
    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelHeader}>
            <Text>Model name</Text>
            <Text>Loss</Text>
            <div />
            <Text>Architecture</Text>
            <div />
            <div />
        </Grid>
    );
};

const TrainJobStatus = ({ job }: { job: SchemaTrainJob }) => {
    if (job.status === 'running') {
        return (
            <View>
                <Flex gap={'size-100'}>
                    <Text UNSAFE_style={{ fontWeight: 500 }}>{job.payload.model_name}</Text>
                    <SplitBadge first={job.status} second={'Fine-tuning the model - epoch n/n'} />
                </Flex>
                {job.start_time ? (
                    <Text UNSAFE_className={classes.modelInfo}>
                        Started: {new Date(job.start_time).toLocaleString()} | Elapsed: {elapsedSince(job.start_time)}
                    </Text>
                ) : (
                    <></>
                )}
            </View>
        );
    } else {
        const color = job.status === 'failed' ? 'var(--spectrum-negative-visual-color)' : 'var(--energy-blue)';
        return (
            <View>
                <Flex gap={'size-100'}>
                    <Text UNSAFE_style={{ fontWeight: 500 }}>{job.payload.model_name}</Text>
                    <SingleBadge color={color} text={job.status} />
                </Flex>
                {job.start_time && job.end_time && (
                    <Text UNSAFE_className={classes.modelInfo}>
                        Elapsed: {durationBetween(job.start_time, job.end_time)}
                    </Text>
                )}
            </View>
        );
    }
};

export const TrainingRow = ({ trainJob, onInterrupt }: { trainJob: SchemaTrainJob; onInterrupt: () => void }) => {
    const loss = trainJob.extra_info && (trainJob.extra_info['train/loss_step'] as number | undefined);

    return (
        <View>
            <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelRow}>
                <TrainJobStatus job={trainJob} />
                <Text>{loss ? loss.toFixed(2) : '...'}</Text>
                <div />
                <Text>{trainJob.payload.policy.toUpperCase()}</Text>
                <View>
                    {trainJob.status === 'running' && (
                        <Button variant='secondary' onPress={onInterrupt}>
                            Interrupt
                        </Button>
                    )}
                </View>
            </Grid>

            {trainJob.status === 'running' && (
                <ProgressBar size='S' UNSAFE_className={classes.progressBar} width={'100%'} value={trainJob.progress} />
            )}
        </View>
    );
};

export const ImportExportRow = ({
    job,
    onInterrupt,
    onDownload,
}: {
    job: SchemaJob;
    onInterrupt: () => void;
    onDownload?: () => void;
}) => {
    const payload = job.payload as { model_name?: string; type?: string };
    const jobLabel = job.type === 'export' ? 'Export' : 'Import';

    return (
        <View>
            <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelRow}>
                <View>
                    <Flex gap={'size-100'}>
                        <Text UNSAFE_style={{ fontWeight: 500 }}>{payload.model_name ?? 'Model'}</Text>
                        {job.status === 'running' ? (
                            <SplitBadge first={job.status} second={job.message} />
                        ) : (
                            <SingleBadge
                                color={
                                    job.status === 'failed'
                                        ? 'var(--spectrum-negative-visual-color)'
                                        : 'var(--energy-blue)'
                                }
                                text={job.status}
                            />
                        )}
                    </Flex>
                </View>
                <Text>{jobLabel}</Text>
                <div />
                <View>
                    <Flex gap='size-100'>
                        {job.status === 'completed' && job.type === 'export' && onDownload && (
                            <Button variant='accent' onPress={onDownload}>
                                Download
                            </Button>
                        )}
                        {(job.status === 'running' || job.status === 'pending') && (
                            <Button variant='secondary' onPress={onInterrupt}>
                                Cancel
                            </Button>
                        )}
                    </Flex>
                </View>
            </Grid>
        </View>
    );
};
