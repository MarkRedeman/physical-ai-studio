import { useCallback, useEffect, useState } from 'react';

import { Button, Flex, Grid, Heading, Slider, View } from '@geti/ui';
import { sortBy } from 'lodash-es';
import useWebSocket from 'react-use-websocket';
import { degToRad, radToDeg } from 'three/src/math/MathUtils.js';

import { $api } from '../../../api/client';
import { useProjectId } from '../../projects/use-project';
import { RobotViewer } from '../controller/robot-viewer';
import { IdentifyRobotButton } from '../identify-robot-button';
import { useRobotModels } from '../robot-models-context';

const POSITIONS = {
    Middle: {
        shoulder_pan: 0,
        shoulder_lift: 0,
        elbow_flex: 0,
        wrist_flex: 0,
        wrist_roll: 0,
        gripper: 0,
    },
    Up: {
        shoulder_pan: 0,
        shoulder_lift: 5,
        elbow_flex: -80,
        wrist_flex: 0,
        wrist_roll: 0,
        gripper: 15,
    },
    Right: {
        shoulder_pan: 85,
    },
    Left: {
        shoulder_pan: -85,
    },
    'Sit up': {
        shoulder_pan: 0,
        shoulder_lift: -80,
        elbow_flex: 10,
        wrist_flex: 80,
        wrist_roll: 50,
        gripper: 15,
    },
    Sit: {
        shoulder_pan: 0,
        shoulder_lift: -80,
        elbow_flex: 90,
        wrist_flex: 20,
        wrist_roll: 50,
        gripper: 15,
    },
    Down: {
        shoulder_pan: 0,
        shoulder_lift: 90,
        elbow_flex: -80,
        wrist_flex: 0,
        wrist_roll: 0,
        gripper: 5,
    },
};

const placeholderJoints = [
    {
        name: 'J1',
        value: 70,
        rangeMin: -360,
        rangeMax: 360,
        decreaseKey: 'q',
        increaseKey: '1',
    },
    {
        name: 'J2',
        value: 20,
        rangeMin: -360,
        rangeMax: 360,
        decreaseKey: '2',
        increaseKey: '2',
    },
    {
        name: 'J3',
        value: 80,
        rangeMin: -360,
        rangeMax: 360,
        decreaseKey: 'e',
        increaseKey: '3',
    },
    {
        name: 'J4',
        value: 60,
        rangeMin: -360,
        rangeMax: 360,
        decreaseKey: 'r',
        increaseKey: '4',
    },
    {
        name: 'J5',
        value: 10,
        rangeMin: -360,
        rangeMax: 360,
        decreaseKey: 't',
        increaseKey: '5',
    },
    {
        name: 'J6',
        value: 84,
        rangeMin: -360,
        rangeMax: 360,
        decreaseKey: 'y',
        increaseKey: '6',
    },
];

const Joint = ({
    name,
    value,
    minValue,
    maxValue,
    isDisabled,
    onChange,
}: {
    name: string;
    value: number;
    minValue: number;
    maxValue: number;
    isDisabled: boolean;
    onChange: (value: number) => void;
}) => {
    const [state, setState] = useState(value);
    useEffect(() => {
        if (isDisabled) {
            setState(value);
        }
    }, [value, isDisabled]);

    return (
        <li>
            <View
                backgroundColor={'gray-50'}
                padding='size-115'
                UNSAFE_style={{
                    //border: '1px solid var(--spectrum-global-color-gray-200)',
                    borderRadius: '4px',
                }}
            >
                <Grid areas={['name value', 'slider slider']} gap='size-100'>
                    <div style={{ gridArea: 'name' }}>
                        <span>{name}</span>
                    </div>
                    <div style={{ gridArea: 'value', display: 'flex', justifyContent: 'end' }}>
                        <span style={{ color: 'var(--energy-blue-light)' }}>{value.toFixed(2)}&deg;</span>
                    </div>
                    <Flex gridArea='slider' gap='size-200'>
                        <Slider
                            aria-label={name}
                            value={state}
                            defaultValue={value}
                            minValue={minValue}
                            maxValue={maxValue}
                            flexGrow={1}
                            isDisabled={isDisabled}
                            onChangeEnd={isDisabled ? undefined : onChange}
                            onChange={setState}
                        />
                    </Flex>
                </Grid>
            </View>
        </li>
    );
};

const useTeleoperateTheRobot = (
    socket: ReturnType<typeof useWebSocket>,
    project_id: string,
    teleoperate_robot_id?: string
) => {
    // WebSocket message handler
    const handleLeaderMessage = useCallback(
        (event: WebSocketEventMap['message']) => {
            try {
                const payload = JSON.parse(event.data);

                if (payload['event'] === 'state_was_updated') {
                    socket.sendJsonMessage({ command: 'set_joints_state', joints: payload['state'] });
                }
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        },
        [socket]
    );

    const mujoco = 'f004cffd-4517-4731-a9ae-c3a041c1fa4c';
    const url =
        teleoperate_robot_id === mujoco
            ? 'ws://localhost:8081'
            : `/api/projects/${project_id}/robots/${teleoperate_robot_id}/ws`;

    const leaderSocket = useWebSocket(url, {
        queryParams: {
            fps: 60,
        },
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleLeaderMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });

    return leaderSocket;
};

type JointsState = Array<{
    name: string;
    value: number;
    rangeMin: number;
    rangeMax: number;
    decreaseKey: string;
    increaseKey: string;
}>;
const useJointState = (project_id: string, robot_id: string, teleoperate_robot_id?: string) => {
    const [isControlled, setIsControlled] = useState(false);
    const [joints, setJoints] = useState<JointsState>([]);
    const { models } = useRobotModels();

    // WebSocket message handler
    const handleMessage = useCallback(
        (event: WebSocketEventMap['message']) => {
            try {
                const payload = JSON.parse(event.data);

                if (payload['event'] === 'state_was_updated') {
                    const newJoints = payload['state'];

                    Object.keys(newJoints).forEach((joint) => {
                        models.forEach((model) => {
                            model.setJointValue(joint, degToRad(newJoints[joint]));
                        });
                    });

                    const modelJoints = Object.values(models.at(0)?.joints ?? {});

                    const jointState = Object.keys(newJoints).map((joint_name, idx) => {
                        const joint = modelJoints.find(({ urdfName }) => urdfName === joint_name);

                        const rangeMax = joint === undefined ? 180 : radToDeg(joint.limit.upper);
                        const rangeMin = joint === undefined ? -180 : radToDeg(joint.limit.lower);

                        return {
                            ...placeholderJoints[idx],
                            name: joint_name,
                            value: Number(newJoints[joint_name]),
                            rangeMax,
                            rangeMin,
                        };
                    });

                    setJoints(jointState);
                    setIsControlled(payload['is_controlled']);
                }
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        },
        [models]
    );

    const mujoco = 'f004cffd-4517-4731-a9ae-c3a041c1fa4c';
    const url = robot_id === mujoco ? 'ws://localhost:8081' : `/api/projects/${project_id}/robots/${robot_id}/ws`;

    const socket = useWebSocket(url, {
        queryParams: {
            fps: 60,
        },
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });

    // WebSocket message handler
    useTeleoperateTheRobot(socket, project_id, teleoperate_robot_id);

    const setJoint = (name: string, value: number) => {
        socket.sendJsonMessage({
            command: 'set_joints_state',
            joints: {
                [name]: value,
            },
        });
    };

    return [joints, isControlled, setJoint, socket] as const;
};

const Identify = ({ robot_id }: { robot_id: string }) => {
    const { project_id } = useProjectId();
    const robot = $api.useQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id } },
    });

    const port_id = robot.data?.serial_id;

    if (port_id === undefined) {
        return null;
    }

    return <IdentifyRobotButton port_id={port_id} />;
};

export const LeaderCell = ({
    robot_id,
    label,
    teleoperate_robot_id,
}: {
    robot_id: string;
    label: string;
    teleoperate_robot_id?: string;
}) => {
    const { project_id } = useProjectId();
    const [joints, isControlled, setJoint, socket] = useJointState(project_id, robot_id, teleoperate_robot_id);
    const isDisabled = isControlled === false;

    return (
        <Grid columns={['2fr', '1fr']} areas={['title title', 'viewer controls']}>
            <View gridArea='title' padding='size-100'>
                <Flex justifyContent={'space-between'}>
                    <Heading level={4}>{label}</Heading>
                    <Identify robot_id={robot_id} />
                </Flex>
            </View>

            <View gridArea='viewer' maxWidth='size-6000' maxHeight='size-6000'>
                <RobotViewer />
            </View>
            <View backgroundColor={'gray-100'} padding='size-50' gridArea='controls'>
                <ul>
                    <Grid gap='size-50' columns={['1fr', '1fr']}>
                        {sortBy(joints, (joint) => joint.name).map((joint) => {
                            return (
                                <Joint
                                    isDisabled={isDisabled}
                                    key={joint.name}
                                    name={joint.name}
                                    value={joint.value}
                                    minValue={joint.rangeMin}
                                    maxValue={joint.rangeMax}
                                    onChange={(value) => {
                                        setJoint(joint.name, value);
                                    }}
                                />
                            );
                        })}
                    </Grid>
                </ul>
                <Grid columns={['1fr', '1fr', '1fr', '1fr']} gap='size-100'>
                    {Object.keys(POSITIONS).map((position) => {
                        return (
                            <Button
                                key={position}
                                width='100%'
                                variant='secondary'
                                onPress={() => {
                                    const payload = POSITIONS[position] ?? {};
                                    socket.sendJsonMessage({
                                        command: 'set_joints_state',
                                        joitns: payload,
                                    });
                                }}
                            >
                                {position}
                            </Button>
                        );
                    })}
                </Grid>
            </View>
        </Grid>
    );
};
