import { Suspense, useState } from 'react';

import {
    Badge,
    Button,
    Content,
    ContextualHelp,
    Flex,
    Footer,
    Grid,
    Heading,
    Icon,
    Item,
    Link,
    Loading,
    minmax,
    Picker,
    Text,
    ToastQueue,
    View,
} from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { AppFooter } from '../../../components/app-footer/app-footer';
import { useDatasetId } from '../../../features/datasets/use-dataset';
import { RuntimeSessionProvider, useRuntimeSession } from '../../../features/robots/runtime-session-provider';
import { paths } from '../../../paths';
import { RecordingViewer } from './recording-viewer';

import classes from './index.module.css';

const TotalRecordedEpisodes = () => {
    const { dataset_id } = useDatasetId();
    const { state } = useRuntimeSession();

    const episodeQuery = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id,
            },
        },
    });
    const totalEpisodes = Number(episodeQuery.data?.length) + state.episodes_recorded;

    if (totalEpisodes === 0 || isNaN(totalEpisodes)) {
        return null;
    }

    return (
        <Flex direction='row' justifyContent={'space-between'} gap='size-150'>
            <Flex alignItems={'center'} gap='size-50'>
                <Text UNSAFE_className={classes.episodesText}>Total episodes recorded</Text>
                <ContextualHelp variant='info'>
                    <Heading>Recommended amount of episodes</Heading>
                    <Content>
                        <Text>We recommend recording at least 50 episodes before training your first model.</Text>
                    </Content>
                    <Footer>
                        <Link href='https://github.com/open-edge-platform/physical-ai-studio/issues/358'>
                            Learn more about recommended dataset sizes
                        </Link>
                    </Footer>
                </ContextualHelp>
            </Flex>
            <div className={classes.episodesCount}>
                <Badge variant='positive'>{totalEpisodes}</Badge>
            </div>
        </Flex>
    );
};

const RecordingPage = () => {
    const { project_id, dataset_id } = useDatasetId();
    const [selectedRecordingEnvironmentId, setSelectedRecordingEnvironmentId] = useState<string>();
    const [environmentToAddId, setEnvironmentToAddId] = useState<string>();

    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: {
            path: {
                dataset_id,
            },
        },
    });

    const { data: recordingEnvironments } = $api.useSuspenseQuery(
        'get',
        '/api/dataset/{dataset_id}/recording-environments',
        { params: { path: { dataset_id } } }
    );
    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });
    const selectedRecordingEnvironment =
        recordingEnvironments.find(({ id }) => id === selectedRecordingEnvironmentId) ?? recordingEnvironments[0];
    const environmentId = selectedRecordingEnvironment?.environment_id;
    const environmentQuery = $api.useQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        { params: { path: { project_id, environment_id: environmentId ?? '' } } },
        { enabled: environmentId !== undefined }
    );
    const createRecordingEnvironment = $api.useMutation('post', '/api/dataset/{dataset_id}/recording-environments', {
        meta: {
            invalidates: [
                ['get', '/api/dataset/{dataset_id}/recording-environments', { params: { path: { dataset_id } } }],
            ],
        },
    });

    const addEnvironment = async () => {
        if (environmentToAddId === undefined) return;
        const recordingEnvironment = await createRecordingEnvironment.mutateAsync({
            params: { path: { dataset_id } },
            body: { environment_id: environmentToAddId, camera_mappings: [] },
        });
        setSelectedRecordingEnvironmentId(recordingEnvironment.id);
    };

    if (selectedRecordingEnvironment === undefined || environmentQuery.data === undefined) {
        return (
            <Flex direction='column' gap='size-200' maxWidth='size-4600' margin='size-400'>
                <Heading>Select a recording environment</Heading>
                {recordingEnvironments.length > 0 && (
                    <Picker
                        label='Recording environment'
                        items={recordingEnvironments}
                        selectedKey={selectedRecordingEnvironmentId ?? recordingEnvironments[0].id}
                        onSelectionChange={(id) => setSelectedRecordingEnvironmentId(id?.toString())}
                    >
                        {(recordingEnvironment) => {
                            const environment = environments.find(
                                ({ id }) => id === recordingEnvironment.environment_id
                            );
                            return (
                                <Item key={recordingEnvironment.id}>
                                    {environment?.name ?? 'Unavailable environment'}
                                </Item>
                            );
                        }}
                    </Picker>
                )}
                <Picker
                    label='Add project environment'
                    items={environments.filter(
                        ({ id }) => !recordingEnvironments.some(({ environment_id }) => environment_id === id)
                    )}
                    selectedKey={environmentToAddId}
                    onSelectionChange={(id) => setEnvironmentToAddId(id?.toString())}
                >
                    {(environment) => <Item key={environment.id}>{environment.name}</Item>}
                </Picker>
                <Button
                    variant='accent'
                    onPress={addEnvironment}
                    isDisabled={environmentToAddId === undefined}
                    isPending={createRecordingEnvironment.isPending}
                >
                    Use environment for recording
                </Button>
            </Flex>
        );
    }
    const environment = environmentQuery.data;
    return (
        <RuntimeSessionProvider environment={environment} dataset={dataset} onError={ToastQueue.negative}>
            <Grid
                areas={['header', 'content', 'footer']}
                UNSAFE_style={{
                    // Footer track matches ProjectLayout's so the strip is the
                    // same height here as on every other page.
                    // eslint-disable-next-line max-len
                    gridTemplateRows: `var(--spectrum-global-dimension-size-800, 4rem) ${minmax(0, '1fr')} var(--spectrum-global-dimension-size-400)`,
                }}
                minHeight={0}
                height={'100%'}
            >
                <View backgroundColor={'gray-300'} gridArea={'header'}>
                    <Flex
                        height='100%'
                        alignItems={'center'}
                        marginX='1rem'
                        gap='size-200'
                        justifyContent={'space-between'}
                    >
                        <Link
                            href={paths.project.datasets.show({ project_id, dataset_id })}
                            isQuiet
                            variant='overBackground'
                        >
                            <Flex marginEnd='size-200' direction='row' gap='size-200' alignItems={'center'}>
                                <Icon>
                                    <ChevronLeft />
                                </Icon>
                                <Flex direction={'column'}>
                                    <Text UNSAFE_className={classes.headerText}>Adding Episode</Text>
                                    <Text UNSAFE_className={classes.subHeaderText}>
                                        {environment.name} | {dataset.default_task}
                                    </Text>
                                </Flex>
                            </Flex>
                        </Link>
                        <TotalRecordedEpisodes />
                    </Flex>
                </View>

                <View gridArea={'content'} maxHeight={'100vh'} minHeight={0} height='100%'>
                    <View padding='size-200' height='100%'>
                        <RecordingViewer />
                    </View>
                </View>
                <AppFooter />
            </Grid>
        </RuntimeSessionProvider>
    );
};

export const Index = () => {
    return (
        <Suspense fallback={<Loading mode='overlay' />}>
            <RecordingPage />
        </Suspense>
    );
};
