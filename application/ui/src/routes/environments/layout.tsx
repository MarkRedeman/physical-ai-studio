import { Suspense } from 'react';

import {
    ActionButton,
    Button,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Grid,
    Heading,
    Icon,
    Item,
    Loading,
    Menu,
    MenuTrigger,
    minmax,
    repeat,
    View,
} from '@geti-ui/ui';
import { Add, MoreMenu } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { NavLink, Outlet, useParams } from 'react-router-dom';

import { $api } from '../../api/client';
import { SchemaEnvironmentOutput } from '../../api/openapi-spec';
import { useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';

import classes from './../../features/robots/robots-list.module.scss';

const MenuActions = ({ environment_id }: { environment_id: string }) => {
    const { project_id } = useProjectId();
    const deleteEnvironmentMutation = $api.useMutation(
        'delete',
        '/api/projects/{project_id}/environments/{environment_id}'
    );

    return (
        <MenuTrigger>
            <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}>
                <MoreMenu />
            </ActionButton>
            <Menu
                selectionMode='single'
                onAction={(action) => {
                    if (action === 'delete') {
                        deleteEnvironmentMutation.mutate({ params: { path: { project_id, environment_id } } });
                    }
                }}
            >
                <Item href={paths.project.environments.edit({ project_id, environment_id })}>Edit</Item>
                <Item key='delete'>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};

const EnvironmentListItemSmall = ({
    environment,
    isActive,
}: {
    environment: SchemaEnvironmentOutput;
    isActive: boolean;
}) => {
    return (
        <View
            padding='size-200'
            UNSAFE_className={clsx({
                [classes.robotListItem]: true,
                [classes.robotListItemActive]: isActive,
            })}
        >
            <Flex direction={'column'} justifyContent={'space-between'} gap={'size-50'}>
                <Grid areas={['name menu']} columns={['auto', '1fr']} gap={'size-100'}>
                    <View gridArea='name'>
                        <Heading level={2} UNSAFE_style={isActive ? { color: 'var(--energy-blue)' } : {}}>
                            {environment.name}
                        </Heading>
                    </View>
                    <View gridArea='menu' alignSelf={'end'} justifySelf={'end'}>
                        <MenuActions environment_id={environment.id} />
                    </View>
                </Grid>
            </Flex>
        </View>
    );
};

const EnvironmentListItem = ({
    environment,
    isActive,
}: {
    environment: SchemaEnvironmentOutput;
    isActive: boolean;
}) => {
    return (
        <View
            padding='size-200'
            UNSAFE_className={clsx({
                [classes.robotListItem]: true,
                [classes.robotListItemActive]: isActive,
            })}
        >
            <Flex direction={'column'} justifyContent={'space-between'} gap={'size-50'}>
                <Grid
                    areas={['icon name status', 'parameters parameters menu']}
                    columns={['auto', '1fr']}
                    gap={'size-100'}
                >
                    <View gridArea={'icon'} padding='size-100'>
                        Env
                    </View>
                    <View gridArea='name'>
                        <Heading level={2} UNSAFE_style={isActive ? { color: 'var(--energy-blue)' } : {}}>
                            {environment.name}
                        </Heading>
                    </View>
                    <View gridArea='status'>...status?</View>
                    <View gridArea='menu' alignSelf={'end'} justifySelf={'end'}>
                        <MenuActions environment_id={environment.id} />
                    </View>
                    <View gridArea='parameters'>
                        <ul
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 'var(--spectrum-global-dimension-size-10)',
                                listStyleType: 'disc',
                                fontSize: '10px',
                            }}
                        >
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                Cameras: {environment.camera_ids?.length ?? 0}
                            </li>
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                Robots: {environment.robots?.length ?? 0}
                            </li>
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                Tele operators:{' '}
                                {environment.robots?.filter(({ tele_operator }) => tele_operator.type === 'robot')
                                    .length ?? 0}
                            </li>
                        </ul>
                    </View>
                </Grid>
            </Flex>
        </View>
    );
};

export const EnvironmentsList = () => {
    const { project_id = '' } = useParams<{ project_id: string }>();
    const {} = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    const environmentsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });

    return (
        <Flex direction='column' gap='size-100'>
            {/* TODO:  */}
            <View isHidden>
                <Flex justifyContent={'space-between'} alignItems={'end'}>
                    <span>Step 3: create an environment</span>
                    <Button>Next</Button>
                </Flex>
                <Divider size='S' marginY='size-200' />
            </View>
            <DialogTrigger type='modal'>
                <Button variant='secondary' UNSAFE_className={classes.addNewRobotButton}>
                    <Icon marginEnd='size-50'>
                        <Add />
                    </Icon>
                    Configure a new environment
                </Button>
                <Dialog>
                    <Heading>Create a new environment</Heading>
                    <Divider />
                    <Content>
                        <Grid gap='size-100' columns={[repeat('auto-fit', minmax('size-3000', '1fr'))]}>
                            <Button
                                variant='secondary'
                                href={paths.project.environments.new({ project_id })}
                                UNSAFE_className={classes.addNewRobotButton}
                                height={'100%'}
                            >
                                Custom
                            </Button>

                            <Button
                                variant='secondary'
                                href={paths.project.environments.new({ project_id })}
                                UNSAFE_className={classes.addNewRobotButton}
                                height={'100%'}
                            >
                                Third Party
                            </Button>

                            <Button
                                variant='secondary'
                                href={paths.project.environments.new({ project_id })}
                                UNSAFE_className={classes.addNewRobotButton}
                                height={'100%'}
                            >
                                SO101
                            </Button>

                            <Button
                                variant='secondary'
                                href={paths.project.environments.new({ project_id })}
                                UNSAFE_className={classes.addNewRobotButton}
                                height={'100%'}
                            >
                                Trossen
                            </Button>

                            <Button
                                variant='secondary'
                                href={paths.project.environments.new({ project_id })}
                                UNSAFE_className={classes.addNewRobotButton}
                                height={'100%'}
                            >
                                LeKiwi
                            </Button>

                            <Button
                                variant='secondary'
                                href={paths.project.environments.new({ project_id })}
                                UNSAFE_className={classes.addNewRobotButton}
                                height={'100%'}
                            >
                                AlohaMini
                            </Button>
                        </Grid>
                    </Content>
                </Dialog>
            </DialogTrigger>

            <Button
                variant='secondary'
                href={paths.project.environments.new({ project_id })}
                UNSAFE_className={classes.addNewRobotButton}
                isHidden
            >
                <Icon marginEnd='size-50'>
                    <Add />
                </Icon>
                Configure a new environment
            </Button>

            <Flex direction='column' gap='size-100'>
                {environmentsQuery.data.map((environment) => {
                    const to = paths.project.environments.show({ project_id, environment_id: environment.id });

                    return (
                        <NavLink key={environment.id} to={to}>
                            {({ isActive }) => {
                                return (
                                    <EnvironmentListItemSmall
                                        environment={environment}
                                        isActive={isActive}
                                        status={'connected'}
                                    />
                                );
                            }}
                        </NavLink>
                    );
                })}
            </Flex>
            <Flex direction='column' gap='size-100' isHidden>
                {environmentsQuery.data.map((environment) => {
                    const to = paths.project.environments.show({ project_id, environment_id: environment.id });

                    return (
                        <NavLink key={environment.id} to={to}>
                            {({ isActive }) => {
                                return (
                                    <EnvironmentListItem
                                        environment={environment}
                                        isActive={isActive}
                                        status={'connected'}
                                    />
                                );
                            }}
                        </NavLink>
                    );
                })}
            </Flex>
        </Flex>
    );
};

export const Layout = () => {
    return (
        <Grid areas={['environments controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
            <View gridArea='environments' backgroundColor={'gray-100'} padding='size-400'>
                <EnvironmentsList />
            </View>
            <View gridArea='controls' backgroundColor={'gray-50'} minHeight={0}>
                <Suspense
                    fallback={
                        <Grid width='100%' height='100%'>
                            <Loading mode='inline' />
                        </Grid>
                    }
                >
                    <Outlet />
                </Suspense>
            </View>
        </Grid>
    );
};
