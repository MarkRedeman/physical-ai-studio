import { useEffect, useState } from 'react';

import { ActionButton, Flex, Grid, Heading, minmax, repeat, Slider, Switch, View } from '@geti/ui';
import { ChevronDownSmallLight } from '@geti/ui/icons';
import { radToDeg } from 'three/src/math/MathUtils.js';

import { urdfPathForType, useRobotModels } from '../robot-models-context';
import { useJointState, useSynchronizeModelJoints } from '../use-joint-state';
import { useRobot, useRobotId } from '../use-robot';

const Joint = ({
    name,
    value,
    minValue,
    maxValue,
    decreaseKey,
    increaseKey,
    isDisabled,
    onChange,
}: {
    name: string;
    value: number;
    minValue: number;
    maxValue: number;
    decreaseKey: string;
    increaseKey: string;
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
            <View backgroundColor={'gray-50'} padding='size-115' UNSAFE_style={{ borderRadius: '4px' }}>
                <Grid areas={['name value', 'slider slider']} gap='size-100'>
                    <div style={{ gridArea: 'name' }}>
                        <span>{name}</span>
                    </div>
                    <div style={{ gridArea: 'value', display: 'flex', justifyContent: 'end' }}>
                        <span style={{ color: 'var(--energy-blue-light)' }}>{value.toFixed(2)}&deg;</span>
                    </div>
                    <Flex gridArea='slider' gap='size-200'>
                        <View
                            isHidden
                            backgroundColor={'gray-100'}
                            paddingY='size-50'
                            paddingX='size-150'
                            UNSAFE_style={{
                                borderRadius: '4px',
                            }}
                        >
                            <kbd>{decreaseKey}</kbd>
                        </View>
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
                        <View
                            isHidden
                            backgroundColor={'gray-100'}
                            paddingY='size-50'
                            paddingX='size-150'
                            UNSAFE_style={{
                                borderRadius: '4px',
                            }}
                        >
                            <kbd>{increaseKey}</kbd>
                        </View>
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
const useJointStateOld = (isEnabled: boolean) => {
    const robot = useRobot();
    const PATH = urdfPathForType(robot.type);
    const { getModel } = useRobotModels();
    const model = getModel(PATH);

    const [isControlled, setIsControlled] = useState(false);

    const { project_id, robot_id } = useRobotId();
    const { joints, socket } = useJointState(project_id, robot_id);
    const modelJoints = Object.values(model?.joints ?? {});

    useSynchronizeModelJoints(joints, PATH);
    const jointsWithRanges: JointsState = joints.map((joint) => {
        const modelJoint = modelJoints.find(({ urdfName }) => urdfName === joint.name);
        const rangeMax = modelJoint === undefined ? 180 : radToDeg(modelJoint.limit.upper);
        const rangeMin = modelJoint === undefined ? -180 : radToDeg(modelJoint.limit.lower);

        return {
            ...joint,
            rangeMin,
            rangeMax,
            decreaseKey: '',
            increaseKey: '',
        };
    });

    const setJoint = (name: string, value: number) => {
        socket.sendJsonMessage({
            command: 'set_joints_state',
            joints: {
                [name]: value,
            },
        });
    };

    return [jointsWithRanges, isControlled, setJoint, socket] as const;
};

const Joints = ({
    joints,
    setJoint,
    isDisabled,
}: {
    joints: JointsState;
    setJoint: (name: string, value: number) => void;
    isDisabled: boolean;
}) => {
    return (
        <ul>
            <Grid gap='size-50' columns={repeat('auto-fit', minmax('size-4600', '1fr'))}>
                {joints.map((joint) => {
                    return (
                        <Joint
                            isDisabled={isDisabled}
                            key={joint.name}
                            name={joint.name}
                            value={joint.value}
                            minValue={joint.rangeMin}
                            maxValue={joint.rangeMax}
                            decreaseKey={joint.decreaseKey}
                            increaseKey={joint.increaseKey}
                            onChange={(value) => {
                                setJoint(joint.name, value);
                            }}
                        />
                    );
                })}
            </Grid>
        </ul>
    );
};

const CompoundMovements = () => {
    return null;
    return (
        <>
            <Heading level={4}>Compound movements</Heading>
            <ul>
                <Flex gap='size-50'>
                    <li>
                        <View
                            backgroundColor={'gray-50'}
                            padding='size-115'
                            UNSAFE_style={{
                                //border: '1px solid var(--spectrum-global-color-gray-200)',
                                borderRadius: '4px',
                            }}
                        >
                            <Flex gap='size-100' alignItems={'center'}>
                                <span>Jaw down & up</span>
                                <View
                                    backgroundColor={'gray-100'}
                                    paddingY='size-50'
                                    paddingX='size-150'
                                    UNSAFE_style={{
                                        borderRadius: '4px',
                                    }}
                                >
                                    <kbd>i</kbd>
                                </View>
                                <View
                                    backgroundColor={'gray-100'}
                                    paddingY='size-50'
                                    paddingX='size-150'
                                    UNSAFE_style={{
                                        borderRadius: '4px',
                                    }}
                                >
                                    <kbd>8</kbd>
                                </View>
                            </Flex>
                        </View>
                    </li>
                    <li>
                        <View
                            backgroundColor={'gray-50'}
                            padding='size-115'
                            UNSAFE_style={{
                                //border: '1px solid var(--spectrum-global-color-gray-200)',
                                borderRadius: '4px',
                            }}
                        >
                            <Flex gap='size-100' alignItems={'center'}>
                                <span>Jaw backward & forward</span>
                                <View
                                    backgroundColor={'gray-100'}
                                    paddingY='size-50'
                                    paddingX='size-150'
                                    UNSAFE_style={{
                                        borderRadius: '4px',
                                    }}
                                >
                                    <kbd>u</kbd>
                                </View>
                                <View
                                    backgroundColor={'gray-100'}
                                    paddingY='size-50'
                                    paddingX='size-150'
                                    UNSAFE_style={{
                                        borderRadius: '4px',
                                    }}
                                >
                                    <kbd>o</kbd>
                                </View>
                            </Flex>
                        </View>
                    </li>
                </Flex>
            </ul>
        </>
    );
};

const Internal = () => {
    const [joints, isControlled, setJoint, socket] = useJointStateOld(true);
    const [expanded, setExpanded] = useState(true);

    return (
        <>
            <Flex justifyContent={'space-between'}>
                <ActionButton onPress={() => setExpanded((c) => !c)}>
                    <Heading level={4} marginX='size-100'>
                        <Flex alignItems='center' gap='size-100'>
                            <ChevronDownSmallLight
                                fill='white'
                                style={{
                                    transform: expanded ? 'rotate(180deg)' : '',
                                    animation: 'transform ease-in-out 0.1s',
                                }}
                            />
                            Joint state
                        </Flex>
                    </Heading>
                </ActionButton>

                <Switch
                    isSelected={isControlled}
                    onChange={(value) => {
                        if (value === true) {
                            socket.sendJsonMessage({ command: 'enable_torque' });
                        } else {
                            socket.sendJsonMessage({ command: 'disable_torque' });
                        }
                    }}
                >
                    Take control
                </Switch>
            </Flex>
            {expanded && (
                <>
                    <Joints joints={joints} isDisabled={isControlled === false} setJoint={setJoint} />
                    <CompoundMovements />
                </>
            )}
        </>
    );
};

const DisabledJointsControls = () => {
    const [expanded, setExpanded] = useState(true);
    const isControlled = false;

    const robot = useRobot();
    const PATH = urdfPathForType(robot.type);
    const { getModel } = useRobotModels();
    const model = getModel(PATH);
    const modelJoints = Object.values(model?.joints ?? {});
    const joints: JointsState = modelJoints
        .filter((joint) => joint.jointType !== 'fixed')
        .map((joint) => {
            const rangeMax = radToDeg(joint.limit.upper);
            const rangeMin = radToDeg(joint.limit.lower);

            return {
                name: joint.name,
                value: 0,
                decreaseKey: '',
                increaseKey: '',
                rangeMin,
                rangeMax,
            };
        })
        .toReversed();

    return (
        <>
            <Flex justifyContent={'space-between'}>
                <ActionButton onPress={() => setExpanded((c) => !c)}>
                    <Heading level={4} marginX='size-100'>
                        <Flex alignItems='center' gap='size-100'>
                            <ChevronDownSmallLight
                                fill='white'
                                style={{
                                    transform: expanded ? 'rotate(180deg)' : '',
                                    animation: 'transform ease-in-out 0.1s',
                                }}
                            />
                            Joint state
                        </Flex>
                    </Heading>
                </ActionButton>

                <Switch isSelected={isControlled} isDisabled>
                    Take control
                </Switch>
            </Flex>
            {expanded && (
                <>
                    <Joints
                        joints={joints}
                        isDisabled={isControlled === false}
                        setJoint={() => {
                            console.info('nono');
                        }}
                    />
                    <CompoundMovements />
                </>
            )}
        </>
    );
};

export const JointControls = ({ isConnected }: { isConnected: boolean }) => {
    if (!isConnected) {
        //return null;
    }

    return (
        <View
            gridArea='controls'
            backgroundColor={'gray-100'}
            padding='size-100'
            UNSAFE_style={{
                border: '1px solid var(--spectrum-global-color-gray-200)',
                borderRadius: '8px',
            }}
        >
            <Flex direction='column' gap='size-50'>
                {isConnected ? <Internal /> : <DisabledJointsControls />}
            </Flex>
        </View>
    );
};
