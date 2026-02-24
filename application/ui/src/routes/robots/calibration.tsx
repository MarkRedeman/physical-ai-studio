import { useCallback, useEffect, useRef, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Cell,
    Column,
    Content,
    Divider,
    Flex,
    Grid,
    Heading,
    InlineAlert,
    Item,
    Menu,
    MenuTrigger,
    Row,
    Slider,
    TableBody,
    TableHeader,
    TableView,
    Text,
    View,
} from '@geti/ui';
import { Button as RacButton } from 'react-aria-components';
import useWebSocket from 'react-use-websocket';
import { degToRad, radToDeg } from 'three/src/math/MathUtils.js';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../api/client';
import { RobotViewer } from '../../features/robots/controller/robot-viewer';
import { RobotModelsProvider, useRobotModels } from '../../features/robots/robot-models-context';
import { useRobot, useRobotId } from '../../features/robots/use-robot';
import RobotArm from './../../assets/robot-arm.png';

import classes from './calibration.module.scss';

const POSITIONS = {
    Center: {
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
                    <div
                        style={{
                            gridArea: 'value',
                            display: 'flex',
                            justifyContent: 'end',
                        }}
                    >
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

type JointsState = Array<{
    name: string;
    value: number;
    rangeMin: number;
    rangeMax: number;
    decreaseKey: string;
    increaseKey: string;
}>;
const useJointState = (normalize = true) => {
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

                    // TODO: instead of relying on the urdf file perhaps we can
                    // rely on the robot calibration's values?
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

    const { project_id, robot_id } = useRobotId();
    const socket = useWebSocket(`/api/projects/${project_id}/robots/${robot_id}/ws`, {
        queryParams: {
            fps: 30,
            normalize,
        },
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });

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

const useNormalizedJointState = () => {
    const [joints, setJoints] = useState<JointsState>([]);

    // WebSocket message handler
    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const payload = JSON.parse(event.data);

            if (payload['event'] === 'state_was_updated') {
                const newJoints = payload['state'];

                setJoints((oldJoints) => {
                    return Object.keys(newJoints).map((joint_name, idx) => {
                        const oldJoint = oldJoints.find((joint) => joint.name === joint_name);
                        const value = Number(newJoints[joint_name]);

                        return {
                            ...placeholderJoints[idx],
                            name: joint_name,
                            value,
                            rangeMax: Math.max(value, oldJoint?.rangeMax ?? 0),
                            rangeMin: Math.min(value, oldJoint?.rangeMin ?? Infinity),
                        };
                    });
                });
            }
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }, []);

    const { project_id, robot_id } = useRobotId();
    useWebSocket(`/api/projects/${project_id}/robots/${robot_id}/ws`, {
        queryParams: {
            fps: 30,
            normalize: 'false',
        },
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });

    return joints;
};

const useCenterRobotModel = () => {
    const { models } = useRobotModels();

    useEffect(() => {
        models.forEach((model) => {
            Object.values(model.joints).forEach((joint) => {
                joint.setJointValue(0);
            });
        });
    }, [models]);
};
const CalibrationCentering = ({ step: _, setStep }: { step: number; setStep: (step: number) => void }) => {
    useCenterRobotModel();
    const state = useNormalizedJointState();

    return (
        <>
            <View gridArea='calibration'>
                <Flex direction='column' gap='size-200'>
                    <Heading level={2}>Centering</Heading>
                    <Text>
                        Move the robot to the center of its range of motion, as shown, and tehn press &ldquo;Capture
                        position&rdquo;
                    </Text>
                    <TableView aria-label='Example table with static contents' gridArea='table'>
                        <TableHeader>
                            <Column>Joint</Column>
                            <Column>Offset</Column>
                        </TableHeader>
                        <TableBody>
                            {state.map((joint) => {
                                return (
                                    <Row key={joint.name}>
                                        <Cell>{joint.name}</Cell>

                                        <Cell>
                                            <CellValue value={joint.value} />
                                        </Cell>
                                    </Row>
                                );
                            })}
                        </TableBody>
                    </TableView>
                    <ButtonGroup>
                        <Button onPress={() => setStep(2)}>Capture position</Button>
                    </ButtonGroup>
                </Flex>
            </View>
        </>
    );
};

const CellValue = ({ value }: { value: number }) => {
    const [prevValue, setPrevValue] = useState(value);
    const [pulse, setPulse] = useState<'increase' | 'decrease' | null>(null);
    const [minValue, setMinValue] = useState(value);
    const [maxValue, setMaxValue] = useState(value);

    const PULSE_THRESHOLD = 5;

    useEffect(() => {
        const delta = Math.abs(value - prevValue);

        if (delta >= PULSE_THRESHOLD) {
            if (value > prevValue) {
                setPulse('increase');
            } else if (value < prevValue) {
                setPulse('decrease');
            }
        }

        setPrevValue(value);

        // Update min/max
        setMinValue((prev) => Math.min(prev, value));
        setMaxValue((prev) => Math.max(prev, value));

        // Reset pulse after animation completes
        const timer = setTimeout(() => setPulse(null), 600);
        return () => clearTimeout(timer);
    }, [value, prevValue]);

    const range = maxValue - minValue || 1;
    const intensity = (value - minValue) / range;

    return (
        <span
            className={pulse === 'increase' ? classes.pulseGreen : pulse === 'decrease' ? classes.pulseRed : ''}
            style={{
                opacity: 0.6 + intensity * 0.4,
                backgroundColor: '#111',
                padding: '.5em',
                borderRadius: '.25em',
                display: 'inline-block',
                fontWeight: '600',
            }}
        >
            {value}
        </span>
    );
};

const CalibrationRange = ({ step: _, setStep }: { step: number; setStep: (step: value) => void }) => {
    useCenterRobotModel();

    const state = useNormalizedJointState();

    return (
        <>
            <View gridArea='calibration'>
                <Flex gap='size-200' direction='column'>
                    <Flex direction='column' gap='size-200'>
                        <Heading level={2}>Range</Heading>
                        <Text>
                            Move each of your robot&apos;s joints through its full range of motion as demonstrated in
                            the video below. The table below captures the minimum and maximum value of your robot&apos;s
                            motors.
                        </Text>
                    </Flex>
                    <TableView aria-label='Example table with static contents' gridArea='table'>
                        <TableHeader>
                            <Column>Joint</Column>
                            <Column>Min</Column>
                            <Column>Current</Column>
                            <Column>Max</Column>
                        </TableHeader>
                        <TableBody>
                            {state.map((joint) => {
                                return (
                                    <Row key={joint.name}>
                                        <Cell>{joint.name}</Cell>
                                        <Cell>
                                            <CellValue value={joint.rangeMin} />
                                        </Cell>

                                        <Cell>
                                            <CellValue value={joint.value} />
                                        </Cell>
                                        <Cell>
                                            <CellValue value={joint.rangeMax} />
                                        </Cell>
                                    </Row>
                                );
                            })}
                        </TableBody>
                    </TableView>
                    <Text>
                        Click the &ldquo;verify calibration&rdquo; button below when the min and max values no longer
                        change while moving the robot.
                    </Text>
                    <ButtonGroup>
                        <Button onPress={() => setStep(2)}>Verify calibration</Button>
                    </ButtonGroup>
                </Flex>
            </View>
        </>
    );
};

const CalibrationVerify = ({ step: _, setStep }: { step: number; setStep: (step: value) => void }) => {
    const [joints, isControlled, setJoint, socket] = useJointState();
    const isDisabled = isControlled === false;

    const [freeMovement, setFreeMovement] = useState(true);

    //
    // To verify we should:
    // 1. store the previous calibration into the =project_robots_calibrations= table, and temporary file
    //    the table should have (id, payload, file, calibration_date, robot_serial_id, is_completed)
    // 2. the robot worker should be able to take a custom calibration file, or calibration id
    // 3. Once verified if the user submits the calibration then we update the robot's =calibration_id=
    // and make sure we store the calibration file properly

    const onNext = () => {
        setFreeMovement(false);
        socket.sendJsonMessage({ command: 'enable_torque' });
    };

    const onPrevious = () => {
        setFreeMovement(true);
        socket.sendJsonMessage({ command: 'disable_torque' });
    };

    return (
        <>
            <View gridArea='calibration'>
                <Flex gap='size-200' direction='column'>
                    <Flex direction='column' gap='size-200'>
                        <Heading level={2}>Verify calibration</Heading>
                        {freeMovement ? (
                            <>
                                <Flex justifyContent={'space-between'}>
                                    <Heading level={3}>Free movement (1/2)</Heading>
                                    <Button variant='secondary' onPress={onNext}>
                                        Next
                                    </Button>
                                </Flex>
                                <Text>
                                    Let&apos;s now verify your calibration. Move the robot into different positions and
                                    verify that the robot seen in the left screen moves along.
                                </Text>
                                <View backgroundColor={'gray-100'} padding='size-50'>
                                    <ul>
                                        <Grid gap='size-50' columns={['1fr', '1fr']}>
                                            {joints.map((joint) => {
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
                                </View>
                            </>
                        ) : (
                            <>
                                <Flex justifyContent={'space-between'}>
                                    <Heading level={3}>Controlled movement (2/2)</Heading>
                                    <Button variant='secondary' onPress={onPrevious}>
                                        Previous
                                    </Button>
                                </Flex>

                                <Text>Next we will assume controlled movement</Text>

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
                                                        joints: payload,
                                                    });
                                                }}
                                            >
                                                {position}
                                            </Button>
                                        );
                                    })}
                                </Grid>

                                <View backgroundColor={'gray-100'} padding='size-50'>
                                    <ul>
                                        <Grid gap='size-50' columns={['1fr', '1fr']}>
                                            {joints.map((joint) => {
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
                                </View>
                            </>
                        )}

                        <Divider size='S' />
                    </Flex>

                    <ButtonGroup>
                        <Button onPress={() => setStep(2)}>Submit calibration</Button>
                    </ButtonGroup>
                </Flex>
            </View>
        </>
    );
};

const MovementPreview = () => {
    const { models } = useRobotModels();

    return (
        <Flex alignItems={'center'} gap='size-200'>
            <Text>TODO: animate these</Text>
            <ButtonGroup>
                {Object.keys(POSITIONS).map((position) => {
                    return (
                        <Button
                            key={position}
                            onPress={() => {
                                models.forEach((model) => {
                                    Object.values(model.joints).forEach((joint) => {
                                        const value = POSITIONS[position][joint.urdfName];

                                        if (value !== undefined) {
                                            joint.setJointValue(degToRad(value));
                                        }
                                    });
                                });
                            }}
                        >
                            {position}
                        </Button>
                    );
                })}
            </ButtonGroup>
        </Flex>
    );
};

export const Calibration = () => {
    const [step, setStep] = useState(0);

    const { project_id, robot_id } = useRobotId();
    const robot = useRobot();
    // const calibrationsQuery = $api.useSuspenseQuery(
    //     'get',
    //     '/api/projects/{project_id}/robots/{robot_id}/calibrations',
    //     {
    //         params: { path: { project_id, robot_id } },
    //     }
    // );

    const motorCalibrationQuery = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/robots/{robot_id}/calibrations/motor',
        {
            params: { path: { project_id, robot_id } },
        }
    );
    const activeCalibrationsQuery = $api.useQuery(
        'get',
        '/api/projects/{project_id}/robots/{robot_id}/calibrations/{calibration_id}',
        {
            params: {
                path: {
                    project_id,
                    robot_id,
                    calibration_id: robot.active_calibration_id ?? '',
                },
            },
        },
        { enabled: robot.active_calibration_id !== null }
    );

    const submitCalibrationMutation = $api.useMutation(
        'post',
        '/api/projects/{project_id}/robots/{robot_id}/calibrations'
    );
    const updateRobotMutation = $api.useMutation('put', '/api/projects/{project_id}/robots/{robot_id}');
    // const updateCalibration = $api.useMutation('put', '/api/projects/{project_id}/robots/{robot_id}/calibrations');

    const joints = Object.keys(motorCalibrationQuery.data);
    const fileInputRef = useRef(null);
    const handleFileImport = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        try {
            const json = await file.text();
            const calibration = JSON.parse(json);
            //const calibrationId = uuidv4();
            //console.log({calibration, calibrationId,});
            alert(calibration);
        } catch (error) {
            console.error('Failed to import calibration:', error);
        }
    };

    return (
        <View paddingY='size-400'>
            <input
                ref={fileInputRef}
                type='file'
                accept='.json'
                onChange={handleFileImport}
                style={{ display: 'none' }}
            />
            <Flex direction='column' gap='size-200' height='100%' maxHeight='100vh'>
                <Grid
                    areas={
                        step < 1
                            ? ['controls controls', 'table table', 'navigation navigation', ' controller calibration']
                            : ['navigation navigation', ' controller calibration']
                    }
                    height='100%'
                    rows={step < 1 ? ['auto', 'auto', 'auto', '1fr'] : ['atuo', '1fr']}
                    columns={['2fr', '1fr']}
                    gap='size-400'
                >
                    {step === 0 ? (
                        <>
                            <View gridArea='controls'>
                                <Flex direction='column' gap='size-200'>
                                    <ButtonGroup>
                                        <Button onPress={() => setStep(1)}>Begin calibration</Button>
                                        <MenuTrigger>
                                            <Button>Import</Button>
                                            <Menu
                                                onAction={async (key) => {
                                                    if (key === 'from_robot') {
                                                        const calibration = Object.fromEntries(
                                                            Object.entries(motorCalibrationQuery.data).map(
                                                                ([key, value]) => [key, { ...value, joint_name: key }]
                                                            )
                                                        );

                                                        const calibrationId = uuidv4();
                                                        await submitCalibrationMutation.mutateAsync({
                                                            params: { path: { project_id, robot_id } },
                                                            body: {
                                                                id: calibrationId,
                                                                file_path: '',
                                                                robot_id,
                                                                values: calibration,
                                                            },
                                                        });
                                                        updateRobotMutation.mutate({
                                                            params: { path: { project_id, robot_id } },
                                                            body: { ...robot, active_calibration_id: calibrationId },
                                                        });
                                                    } else {
                                                        // TODO: read calibration from uplaoded file,
                                                        // add error handling etc
                                                        fileInputRef.current?.click();
                                                    }
                                                }}
                                            >
                                                <Item key='from_robot'>From robot</Item>
                                                <Item key='from_file'>From file</Item>
                                            </Menu>
                                        </MenuTrigger>

                                        <MenuTrigger>
                                            <Button>Export</Button>
                                            <Menu
                                                onAction={(key) => {
                                                    const data =
                                                        key === 'from_robot'
                                                            ? motorCalibrationQuery.data
                                                            : activeCalibrationsQuery.data;

                                                    const json = JSON.stringify(data, null, 2);
                                                    const blob = new Blob([json], {
                                                        type: 'application/json',
                                                    });
                                                    const url = URL.createObjectURL(blob);
                                                    const a = document.createElement('a');
                                                    a.href = url;
                                                    a.download = 'geti-action-robot-calibration-state.json';
                                                    a.click();
                                                    URL.revokeObjectURL(url);
                                                }}
                                                disabledKeys={robot.calibration_id === null ? ['from_action'] : []}
                                            >
                                                <Item key='from_robot'>From robot</Item>
                                                <Item key='from_action'>From Geti Action</Item>
                                            </Menu>
                                        </MenuTrigger>
                                    </ButtonGroup>
                                    {robot.calibration_id === null && (
                                        <InlineAlert variant='notice'>
                                            <Heading>Calibration required</Heading>
                                            <Content>
                                                Calibrate your robot using Geti Action. If you&apos;ve recently
                                                calibrated your robot with an external tool then you may import your
                                                calibration settings from the robot. Otherwise start the calibration
                                                process.
                                            </Content>
                                        </InlineAlert>
                                    )}
                                </Flex>
                            </View>
                            <TableView
                                aria-label='Example table with static contents'
                                isHidden={step > 0}
                                gridArea='table'
                                selectionMode='none'
                            >
                                <TableHeader>
                                    <Column isRowHeader showDivider>
                                        Joint
                                    </Column>
                                    <Column isRowHeader title='Motor'>
                                        <Column>Min</Column>
                                        <Column>Offset</Column>
                                        <Column showDivider>Max</Column>
                                    </Column>
                                    <Column isRowHeader title='Geti Action'>
                                        <Column>Min</Column>
                                        <Column>Offset</Column>
                                        <Column showDivider>Max</Column>
                                    </Column>
                                </TableHeader>
                                <TableBody>
                                    {joints.map((jointName) => {
                                        const motor = motorCalibrationQuery.data[jointName];
                                        const database = activeCalibrationsQuery.data?.values[jointName];

                                        return (
                                            <Row key={jointName}>
                                                <Cell>{jointName}</Cell>

                                                <Cell>{motor.range_min}</Cell>
                                                <Cell>{motor.range_max}</Cell>
                                                <Cell>{motor.homing_offset}</Cell>

                                                <Cell>{database?.range_min}</Cell>
                                                <Cell>{database?.range_max}</Cell>
                                                <Cell>{database?.homing_offset}</Cell>
                                            </Row>
                                        );
                                    })}
                                </TableBody>
                            </TableView>
                        </>
                    ) : null}

                    {step > 0 && <NavigationDuringCalibration step={step} setStep={setStep} />}

                    <RobotModelsProvider>
                        {step > 0 && (
                            <View gridArea='controller'>
                                <Flex gap='size-200' direction='column'>
                                    <View
                                        width={'100%'}
                                        height={'size-6000'}
                                        padding='size-300'
                                        borderWidth='thin'
                                        borderColor='gray-100'
                                    >
                                        <RobotViewer />
                                    </View>

                                    {step === 2 && <MovementPreview />}
                                </Flex>
                            </View>
                        )}
                        {step === 1 && <CalibrationCentering step={step} setStep={setStep} />}

                        {step === 2 && <CalibrationRange step={step} setStep={setStep} />}
                        {step === 3 && <CalibrationVerify step={step} setStep={setStep} />}
                    </RobotModelsProvider>
                </Grid>
            </Flex>
        </View>
    );
};

const NavigationItem = ({
    isActive = false,
    activeStep,
    step,
    label,
    onPress,
}: {
    isActive?: boolean;
    activeStep: number;
    step: number;
    label: string;
    onPress: () => void;
}) => {
    const style =
        isActive || activeStep >= step
            ? {
                  borderRadius: '50%',
                  backgroundColor: 'var(--energy-blue)',
                  color: 'black',
                  width: '1.5rem',
                  height: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold',
                  fontSize: '0.9rem',
              }
            : {
                  borderRadius: '50%',
                  backgroundColor: 'var(--spectrum-global-color-gray-200)',
                  color: 'var(--spectrum-global-color-gray-400)',
                  width: '1rem',
                  height: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold',
                  fontSize: '0.5rem',
              };
    return (
        <RacButton onPress={onPress} style={{ border: 'none', padding: 0, background: 'none' }}>
            <View
                backgroundColor={'gray-100'}
                height='size-600'
                width='size-1600'
                borderRadius={'small'}
                padding='size-100'
                borderBottomColor={isActive ? 'static-blue-300' : 'default'}
                borderBottomWidth={'thicker'}
            >
                <Heading level={4}>
                    <Flex alignItems={'center'} gap='size-100'>
                        <div style={style}>{activeStep > step ? '✔' : step}</div>
                        <span>{label}</span>
                    </Flex>
                </Heading>

                <Flex height='100%' justifyContent={'center'} alignItems={'center'} isHidden>
                    <img src={RobotArm} style={{ maxWidth: '40px' }} alt='Robot arm icon' />
                </Flex>
            </View>
        </RacButton>
    );
};

const NavigationDuringCalibration = ({ step, setStep }: { step: number; setStep: (step: number) => void }) => {
    return (
        <View gridArea='navigation'>
            <Flex justifyContent='space-between'>
                <Flex gap='size-200'>
                    <NavigationItem
                        activeStep={step}
                        isActive={step === 1}
                        step={1}
                        label='Centering'
                        onPress={() => setStep(1)}
                    />
                    <NavigationItem
                        activeStep={step}
                        isActive={step === 2}
                        step={2}
                        label='Range'
                        onPress={() => setStep(2)}
                    />
                    <NavigationItem
                        activeStep={step}
                        isActive={step === 3}
                        step={3}
                        label='Verify'
                        onPress={() => setStep(3)}
                    />
                </Flex>
                <ButtonGroup>
                    <Button variant='negative' onPress={() => setStep(0)}>
                        Cancel Calibration
                    </Button>
                </ButtonGroup>
            </Flex>
        </View>
    );
};
