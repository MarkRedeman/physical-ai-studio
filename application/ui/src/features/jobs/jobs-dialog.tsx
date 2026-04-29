import { useMemo, useState } from 'react';

import {
    ActionButton,
    ActionMenu,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Grid,
    Heading,
    Icon,
    Item,
    Link,
    Loading,
    TabList,
    TabPanels,
    Tabs,
    Text,
    useDateFormatter,
    View,
} from '@geti-ui/ui';
import { ChevronDownSmallLight, ChevronUpSmallLight, Delete, DeleteOutline } from '@geti-ui/ui/icons';
import { countBy } from 'lodash-es';

import { $api } from '../../api/client';
import { SchemaDatasetImportJob, SchemaTrainJob } from '../../api/openapi-spec';
import { ImportJobRow } from '../../routes/datasets/import/dataset-import-jobs.component';
import { TrainingRow } from '../../routes/models/job-table.component';
import { useProjectId } from '../projects/use-project';
import { NumberBadge } from './number-badge';

import classes from './number-badge.module.css';

type SchemaJob = SchemaDatasetImportJob | SchemaTrainJob;

const JobRow = ({ job }: { job: SchemaJob }) => {
    if (job.type === 'training') {
        return (
            <TrainingRow
                key={job.id}
                trainJob={job}
                onInterrupt={() => console.log(job)}
                onViewLogs={() => console.log(job)}
            />
        );
    }

    if (job.type === 'dataset_import') {
        return <ImportJobRow key={job.id} job={job} onViewLogs={() => console.log(job.id)} />;
    }
};

const JobsHeading = ({ job }: { job: SchemaJob }) => {
    if (job.type === 'training') {
        return (
            <Flex gap='size-100'>
                <Heading level={4}>Model training</Heading>
                <Text>🞄</Text>
                <Text>{job.payload.model_name}</Text>
            </Flex>
        );
    }

    if (job.type === 'dataset_import') {
        return <Heading level={4}>Dataset import</Heading>;
    }

    return <Heading level={4}>{job.type}</Heading>;
};

const JobActions = ({ job }: { job: SchemaJob }) => {
    // Cancel:
    // UNSAFE_style={{
    //     fill: 'var(--spectrum-global-color-negative)',
    // }}
    return (
        <ActionButton isQuiet>
            <Icon>
                <Delete />
            </Icon>
        </ActionButton>
    );
    return (
        <ActionMenu isQuiet>
            <Item key='edit'>Edit</Item>
            <Item key='duplicate'>Duplicate</Item>
            <Item key='delete'>Delete</Item>
        </ActionMenu>
    );
};

const JobCreatedAt = ({ job }: { job: SchemaJob }) => {
    const formatter = useDateFormatter({
        dateStyle: 'medium',
        timeStyle: 'short',
    });
    if (!job.created_at) {
        return null;
    }

    const date = new Date(job.created_at);

    return <View UNSAFE_className={classes.createdAt}>Created: {formatter.format(date)}</View>;
};

const JobContent = ({ job }: { job: SchemaJob }) => {
    if (job.type === 'training') {
        return (
            <Flex direction='column' gap='size-100'>
                <Text UNSAFE_className={classes.message}>{job.message}</Text>
                <Flex gap='size-100' wrap UNSAFE_className={classes.content}>
                    <Text>Policy: {job.payload.policy}</Text>
                    <Text>🞄</Text>
                    <Text>Max steps{job.payload.max_steps}</Text>
                    <Text>🞄</Text>
                    <Text>Batch Size: {job.payload.auto_scale_batch_size ? 'Auto' : job.payload.batch_size}</Text>
                    <Text>🞄</Text>
                    <Text>Workers: {job.payload.num_workers}</Text>
                    <Text>🞄</Text>
                    <Text>Val split: {job.payload.val_split}</Text>
                    {1 + 1 > 3 && (
                        <>
                            <Text>🞄</Text>
                            <Text>Dataset: {job.payload.dataset_id}</Text>
                            {job.payload.base_model_id && (
                                <>
                                    <Text>🞄</Text>
                                    <Text>Base Model: {job.payload.base_model_id}</Text>
                                </>
                            )}
                        </>
                    )}
                </Flex>
            </Flex>
        );
    }

    // if (job.type === 'dataset_import') {
    //     return <Text UNSAFE_className={classes.message}>{job.message}</Text>;
    // }

    return <Text UNSAFE_className={classes.message}>{job.message}</Text>;
};

const JobWarning = () => {
    return (
        <>
            <Flex alignItems={'center'} justifyContent={'end'} flexGrow={1} flexShrink={1}>
                {jobWarning !== undefined ? (
                    <TooltipTrigger placement={'bottom'}>
                        <PressableElement UNSAFE_style={{ display: 'flex' }}>
                            <Alert aria-label={'Job step alert'} className={classes.warningSectionIcon} />
                        </PressableElement>
                        <Tooltip>{jobWarning.warning?.trim()}</Tooltip>
                    </TooltipTrigger>
                ) : null}

                {!jobStepsExpanded && stepToDisplay?.progress ? (
                    <Text id={`job-scheduler-${job.id}-progress`} data-testid={`job-scheduler-${job.id}-progress`}>
                        {getStepProgress(stepToDisplay.progress)}
                    </Text>
                ) : null}
            </Flex>
        </>
    );
};

const JobSteps = ({ job }: { job: SchemaJob }) => {
    const [jobStepsExpanded, setJobStepsExpanded] = useState(true);

    const jobDuration = '10s';
    const stepToDisplay = {
        stepName: 'test',
        index: 0,
    };

    const onExpandHandler = () => {
        setJobStepsExpanded((expanded) => !expanded);
    };

    job.steps = [{}];

    return (
        <>
            <Divider size='S' orientation='horizontal' />

            <Flex paddingX='size-250' gap={'size-100'} width='100%' justifyContent={'space-between'}>
                {!jobStepsExpanded && stepToDisplay !== undefined && (
                    <Flex flexGrow={1} flexShrink={0} alignItems='center' gap='size-100'>
                        <Loading mode='inline' size='S' />
                        <Text>{`${stepToDisplay.stepName} (${stepToDisplay.index} of ${job.steps.length})`}</Text>
                    </Flex>
                )}
                {jobDuration !== undefined && (
                    <Flex alignItems={'center'}>
                        <Text UNSAFE_className={classes.message}>Completed in: {jobDuration}</Text>
                    </Flex>
                )}
                {job.steps.length > 0 && (
                    <Flex justifyContent={'space-between'} alignItems={'center'}>
                        <ActionButton
                            isQuiet
                            id={`job-scheduler-${job.id}-action-expand`}
                            data-testid={`job-scheduler-${job.id}-action-expand`}
                            onPress={onExpandHandler}
                            aria-expanded={jobStepsExpanded}
                        >
                            <Icon>
                                {jobStepsExpanded ? (
                                    <ChevronDownSmallLight width={24} height={24} />
                                ) : (
                                    <ChevronUpSmallLight width={24} height={24} />
                                )}
                            </Icon>
                        </ActionButton>
                    </Flex>
                )}
            </Flex>
        </>
    );
    return null;
    return (
        <>
            <Divider size='S' orientation='horizontal' />
            <View>
                <Flex direction='column' gap='size-25'>
                    <View paddingX='size-100' paddingY='size-50' backgroundColor={'gray-50'} borderRadius={'small'}>
                        Step 1
                    </View>
                    <View paddingX='size-100' paddingY='size-50' backgroundColor={'gray-50'} borderRadius={'small'}>
                        Step 1
                    </View>
                    <View paddingX='size-100' paddingY='size-50' backgroundColor={'gray-50'} borderRadius={'small'}>
                        Step 1
                    </View>
                </Flex>
            </View>
        </>
    );
};

const JobsList = ({ jobs }: { jobs: Array<SchemaJob> }) => {
    return (
        <Flex direction={'column'} gap='size-100'>
            {jobs.map((job) => {
                return (
                    <View backgroundColor={'gray-75'} padding='size-200' borderRadius={'small'}>
                        <Flex direction='column' gap='size-100'>
                            <Grid
                                areas={['job_type  job_action', 'job_content job_created_at']}
                                width='100%'
                                // columns={['1fr', '1fr']}
                                // rows={['1fr', '1fr']}
                                gap='size-100'
                            >
                                <View gridArea='job_type'>
                                    <JobsHeading job={job} />
                                </View>
                                <View gridArea='job_content'>
                                    <JobContent job={job} />
                                </View>
                                <View gridArea='job_created_at' justifySelf={'end'} alignSelf={'end'}>
                                    <JobCreatedAt job={job} />
                                </View>
                                <View gridArea='job_action' justifySelf={'end'} alignSelf={'start'}>
                                    <JobActions job={job} />
                                </View>
                            </Grid>
                            <JobSteps job={job} />
                        </Flex>
                        {/*
                            <JobRow key={job.id} job={job} />
                              */}
                    </View>
                );
            })}
        </Flex>
    );
};

const JobsFiltering = () => {
    return <View>Hoi</View>;
};

const StatusTab = ({
    children,
    counts,
    isSelected = false,
}: {
    children: string;
    counts: number;
    isSelected: boolean;
}) => {
    return (
        <Flex alignItems={'center'} gap='size-100'>
            <span>{children}</span>

            {counts > 0 && <span className={classes.badge}>{counts}</span>}
        </Flex>
    );
};

export const JobsDialog = () => {
    const { project_id } = useProjectId();
    const [listVisible, setListVisible] = useState(true);
    const [logsSourceId, setLogsSourceId] = useState<string | undefined>(undefined);
    const jobsQuery = $api.useQuery(
        'get',
        '/api/jobs',
        {
            params: undefined,
        },
        {
            refetchInterval: 2000,
        }
    );

    const jobs = (jobsQuery.data ?? [])
        .toSorted((a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime())
        .filter((job) => true || job.project_id === project_id);

    console.log(jobsQuery, jobs);
    const counts = countBy(jobs, (job) => job.status);

    return (
        <Dialog
            width='unset'
            UNSAFE_style={{ background: 'var(--spectrum-global-color-gray-50)' }}
            maxWidth={'min(985px, 95vw)'}
            minWidth='min(985px, 95vw)'
        >
            <Content>
                {1 + 1 > 2 && (
                    <Flex alignItems='center' marginBottom={'size-150'}>
                        <JobsFiltering />
                    </Flex>
                )}

                <Tabs
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '2px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                    minHeight={'size-6000'}
                    height={'calc(100% - var(--spectrum-global-dimension-size-6000))'}
                >
                    <TabList UNSAFE_className={classes.tab}>
                        <Item key='all'>
                            <StatusTab counts={Object.values(counts).reduce((x, y) => x + y, 0) ?? 0}>
                                All jobs
                            </StatusTab>
                        </Item>
                        <Item key='running'>
                            <StatusTab counts={counts['running'] ?? 0}>Runninb jobs</StatusTab>
                        </Item>
                        <Item key='completed'>
                            <StatusTab counts={counts['completed'] ?? 0}>Finished jobs</StatusTab>
                        </Item>
                        <Item key='scheduled'>
                            <StatusTab counts={counts['scheduled'] ?? 0}>Scheduled jobs</StatusTab>
                        </Item>
                        <Item key='canceled'>
                            <StatusTab counts={counts['canceled'] ?? 0}>Cancelled jobs</StatusTab>
                        </Item>
                        <Item key='failed'>
                            <StatusTab counts={counts['failed'] ?? 0}>Failed jobs</StatusTab>
                        </Item>
                    </TabList>
                    <TabPanels marginTop='size-100'>
                        <Item key='all'>
                            <JobsList jobs={jobs} />
                        </Item>
                        <Item key='running'>
                            <JobsList jobs={jobs.filter((job) => job.status === 'running')} />
                        </Item>
                        <Item key='completed'>
                            <JobsList jobs={jobs.filter((job) => job.status === 'completed')} />
                        </Item>
                        <Item key='scheduled'>
                            <JobsList jobs={jobs.filter((job) => job.status === 'pending')} />
                        </Item>
                        <Item key='canceled'>
                            <JobsList jobs={jobs.filter((job) => job.status === 'canceled')} />
                        </Item>
                        <Item key='failed'>
                            <JobsList jobs={jobs.filter((job) => job.status === 'failed')} />
                        </Item>
                    </TabPanels>
                </Tabs>
            </Content>
        </Dialog>
    );
};
