import { Suspense, useEffect, useRef } from 'react';

import { Content, Flex, Heading, IllustratedMessage, Loading, Text, View } from '@geti/ui';
import { DockviewReact, GridviewApi, IGridviewPanelProps } from 'dockview';
import { GridviewReadyEvent } from 'dockview-react';

import { $api } from '../../../api/client';
import { SchemaCameraConfigInput, SchemaWebcamCameraOutput } from '../../../api/openapi-spec';
import { WebRTCConnectionProvider } from '../../../components/stream/web-rtc-connection-provider';
import { CameraView } from '../../../routes/cameras/camera';
import { WebsocketCamera } from '../../cameras/websocket-camera';
import { useProjectId } from '../../projects/use-project';
import { RobotModelsProvider } from '../robot-models-context';
import { ReactComponent as RobotIllustration } from './../../../assets/illustrations/INTEL_08_NO-TESTS.svg';
import { LeaderCell } from './preview-robot-move';
import { useEnvironmentForm } from './provider';

import classes from './form.module.scss';

const EmptyPreview = () => {
    return (
        <IllustratedMessage>
            <RobotIllustration />

            <Flex direction='column' gap='size-200'>
                <Content>
                    <Text>
                        Choose the robots and cameras you&apos; like to add using the form on the left. After connecting
                        the robots and cameras, the preview will appear here.
                    </Text>
                </Content>
                <Heading>Setup your new environment</Heading>
            </Flex>
        </IllustratedMessage>
    );
};

const So101FollowerCell = ({ robot_id, teleoperate_robot_id }: { robot_id: string; teleoperate_robot_id?: string }) => {
    // TODO: identify button
    return (
        <RobotModelsProvider>
            <LeaderCell robot_id={robot_id} label={'Follower'} teleoperate_robot_id={teleoperate_robot_id} />
        </RobotModelsProvider>
    );
};

const So101LeaderCell = ({ robot_id }: { robot_id: string }) => {
    return <span>TODO: revert once we can share websockets...</span>;
    return (
        <RobotModelsProvider>
            <LeaderCell robot_id={robot_id} label={'Leader'} />
        </RobotModelsProvider>
    );
};

const CameraCell = ({
    camera,
    storedCamera,
}: {
    camera: SchemaCameraConfigInput;
    storedCamera: SchemaWebcamCameraOutput;
}) => {
    return (
        <View backgroundColor={'gray-500'}>
            <View maxHeight='100%' height='100%' position='relative'>
                <WebsocketCamera
                    camera={{
                        driver: storedCamera.driver,
                        fingerprint: storedCamera.fingerprint,
                        name: storedCamera.name,
                        hardware_name: storedCamera.hardware_name,
                        fps: storedCamera.payload.fps,
                        width: storedCamera.payload.width,
                        height: storedCamera.payload.height,
                        payload: storedCamera.payload,
                    }}
                />
            </View>
        </View>
    );
};

const CameraCellById = ({ camera_id }: { camera_id: string }) => {
    const availableCamerasQuery = $api.useSuspenseQuery('get', '/api/hardware/cameras');

    const { project_id } = useProjectId();
    const camerasQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    const storedCamera = camerasQuery.data.find((camera) => camera.id === camera_id);
    const availableCamera =
        availableCamerasQuery.data.find(({ driver, fingerprint }) => {
            return storedCamera?.fingerprint === fingerprint && storedCamera.driver === driver;
        }) ?? availableCamerasQuery.data.at(0);

    if (availableCamera === undefined || storedCamera === undefined) {
        return 'loading?';
    }

    return <CameraCell camera={availableCamera} storedCamera={storedCamera} />;
};

const components = {
    leader: (props: IGridviewPanelProps<{ title: string; robot_id: string }>) => {
        return <So101LeaderCell robot_id={props.params.robot_id} />;
    },
    follower: (props: IGridviewPanelProps<{ title: string; robot_id: string; teleoperate_robot_id?: string }>) => {
        return (
            <So101FollowerCell
                robot_id={props.params.robot_id}
                teleoperate_robot_id={props.params.teleoperate_robot_id}
            />
        );
    },
    camera: (props: IGridviewPanelProps<{ camera_id: string }>) => {
        return <CameraCellById camera_id={props.params.camera_id} />;
    },
    default: (props: IGridviewPanelProps<{ title: string }>) => {
        return <div style={{ padding: '20px', color: 'white' }}>{props.params.title}</div>;
    },
};

const ActualPreview = () => {
    const environment = useEnvironmentForm();
    const availableCamerasQuery = $api.useSuspenseQuery('get', '/api/hardware/cameras');

    const { project_id } = useProjectId();
    const robotsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });
    const camerasQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    const robots = environment.robots
        .map((robot) => {
            return {
                availableRobot: robotsQuery.data.find(({ id }) => id === robot.robot_id),
                robot,
            };
        })
        .filter((x) => x.robot !== undefined);
    const teleoperators = environment.robots
        .map(({ teleoperator }) => {
            if (teleoperator.type === 'robot') {
                return robotsQuery.data.find(({ id }) => id === teleoperator.robot_id);
            }

            return undefined;
        })
        .filter((x) => x !== undefined);

    const cameras = environment.camera_ids
        .map((cameraId) => {
            const storedCamera = camerasQuery.data.find((camera) => camera.id === cameraId);
            const availableCamera =
                availableCamerasQuery.data.find(({ driver, fingerprint }) => {
                    return storedCamera?.fingerprint === fingerprint && storedCamera.driver === driver;
                }) ?? availableCamerasQuery.data.at(0);

            if (availableCamera === undefined) {
                return undefined;
            }

            return { availableCamera, storedCamera };
        })
        .filter((x) => x !== undefined);

    const apii = useRef<GridviewApi>();
    const onReady = (event: GridviewReadyEvent) => {
        let lastRobot = '';
        robots.forEach(({ availableRobot, robot }) => {
            if (availableRobot === undefined) {
                return;
            }

            const robotId = availableRobot.name;
            const teleoperateRobotId = robot.teleoperator.robot_id;
            event.api.addPanel({
                id: robotId,
                params: { title: 'Follower', robot_id: availableRobot.id, teleoperate_robot_id: teleoperateRobotId },
                component: 'follower',
            });
            lastRobot = robotId;
        });

        teleoperators.forEach((robot) => {
            if (robot === undefined) {
                return;
            }
            const robotId = robot.name;
            event.api.addPanel({
                id: robotId,
                params: {
                    title: 'Leader',
                    robot_id: robot.id,
                },
                component: 'leader',
            });
        });

        let lastCamera = '';
        cameras.forEach(({ storedCamera }, idx) => {
            const camera_id = storedCamera.id;

            event.api.addPanel({
                id: storedCamera.name,
                component: 'camera',
                params: {
                    title: `Camera ${idx}`,
                    camera_id,
                },
                position:
                    lastCamera !== ''
                        ? { direction: 'right', referencePanel: lastCamera }
                        : lastRobot !== ''
                          ? { direction: 'above', referencePanel: lastRobot }
                          : undefined,
            });
            lastCamera = storedCamera.name;
        });

        apii.current = event.api;
    };

    useEffect(() => {
        if (apii.current === undefined) {
            return;
        }

        const api = apii.current;

        let lastRobot = '';
        robots.forEach(({ robot }) => {
            if (robot === undefined) {
                return;
            }
            const robotId = robot.name;

            if (api.getPanel(robotId) === undefined) {
                api.addPanel({
                    id: robotId,
                    params: { title: 'Follower', robot_id: robot.id },
                    component: 'follower',
                });
            }
            lastRobot = robotId;
        });

        teleoperators.forEach((robot) => {
            if (robot === undefined) {
                return;
            }
            const robotId = robot.name;
            if (api.getPanel(robotId) === undefined) {
                api.addPanel({
                    id: robotId,
                    params: {
                        title: 'Leader',
                        robot_id: robot.id,
                    },
                    component: 'leader',
                });
            }
        });

        let lastCamera = '';
        cameras.forEach(({ storedCamera }, idx) => {
            const camera_id = storedCamera.id;

            if (api.getPanel(storedCamera.name) === undefined) {
                api.addPanel({
                    id: storedCamera.name,
                    component: 'camera',
                    params: {
                        title: `Camera ${idx}`,
                        camera_id,
                    },
                    position:
                        lastCamera !== ''
                            ? { direction: 'right', referencePanel: lastCamera }
                            : lastRobot !== ''
                              ? { direction: 'above', referencePanel: lastRobot }
                              : undefined,
                });
            }

            lastCamera = storedCamera.name;
        });
    }, [cameras, robots, teleoperators]);

    return (
        <DockviewReact
            onReady={onReady}
            components={components}
            theme={{
                className: classes.dockview,
            }}
        />
    );
};

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const Preview = () => {
    const environment = useEnvironmentForm();

    const hasRobots = environment.robots.length > 0;
    const hasCameras = environment.camera_ids.length > 0;

    if (hasRobots || hasCameras) {
        return (
            <View UNSAFE_className={classes.dockview} height='100%'>
                <Suspense fallback={<CenteredLoading />}>
                    <ActualPreview />
                </Suspense>
            </View>
        );
    }

    return (
        <View
            backgroundColor={'gray-200'}
            height={'100%'}
            maxHeight='100vh'
            padding='size-200'
            UNSAFE_style={{
                borderRadius: 'var(--spectrum-alias-border-radius-regular)',
                borderColor: 'var(--spectrum-global-color-gray-700)',
                borderWidth: '1px',
                borderStyle: 'dashed',
            }}
            position={'relative'}
        >
            {hasRobots || hasCameras ? (
                <Suspense fallback={<CenteredLoading />}>
                    <ActualPreview />
                </Suspense>
            ) : (
                <EmptyPreview />
            )}
        </View>
    );
};
