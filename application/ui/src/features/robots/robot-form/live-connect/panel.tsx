import { useEffect, useState } from 'react';

import {
    AlertDialog,
    Button,
    DialogContainer,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Heading,
    Slider,
    Text,
    View,
} from '@geti-ui/ui';
import { radToDeg } from 'three/src/math/MathUtils.js';
import { URDFRobot } from 'urdf-loader';

import { useRobotCatalogDefinitionQuery } from '../../robot-catalog.hooks';
import { useLoadModelQuery } from '../../robot-models-context';
import { useRobotForm } from '../provider';
import { useLiveConnect, useLiveJoints } from './provider';

export const jointPresentation = (
    type: string,
    unit: unknown,
    name: string,
    value: number,
    model: URDFRobot | undefined,
    jointMap: Record<string, string[]>
) => {
    const joint = model?.joints[jointMap[name]?.[0]];
    const isSO101 = type.startsWith('SO101_') || type.startsWith('BimanualSO101_');
    const isTrossen = type.startsWith('Trossen_');
    if (joint === undefined || (!isSO101 && !isTrossen)) return { value, unit: 'raw', range: null, modelValue: null };
    const angular = joint.jointType === 'revolute' || joint.jointType === 'continuous';
    if (isSO101 && unit !== 'degrees') {
        const range = name.includes('gripper') ? ([0, 100] as const) : ([-100, 100] as const);
        const modelValue =
            joint.limit.lower + ((value - range[0]) / (range[1] - range[0])) * (joint.limit.upper - joint.limit.lower);
        return { value, unit: 'normalized', range, modelValue: Number.isFinite(modelValue) ? modelValue : null };
    }
    const range =
        joint.jointType === 'continuous' ||
        !Number.isFinite(joint.limit.lower) ||
        !Number.isFinite(joint.limit.upper) ||
        joint.limit.upper <= joint.limit.lower
            ? null
            : ([
                  angular ? radToDeg(joint.limit.lower) : joint.limit.lower,
                  angular ? radToDeg(joint.limit.upper) : joint.limit.upper,
              ] as const);
    return { value, unit: angular ? '°' : 'm', range, modelValue: angular ? (value * Math.PI) / 180 : value };
};

export const LiveJointStates = ({ robotType }: { robotType: string }) => {
    const { payload } = useRobotForm();
    const joints = useLiveJoints();
    const { data: definition } = useRobotCatalogDefinitionQuery(robotType);
    const { data: model } = useLoadModelQuery(robotType);
    const readings = Object.entries(joints).map(([name, value]) => ({
        name,
        ...jointPresentation(robotType, payload.unit, name, value, model, definition.joint_map),
    }));

    useEffect(() => {
        if (model === undefined) return;
        if (Object.keys(joints).length === 0) {
            Object.keys(model.joints).forEach((name) => model.setJointValue(name, 0));
            return;
        }
        for (const [name, value] of Object.entries(joints)) {
            for (const target of definition.joint_map[name] ?? []) {
                const mapped = jointPresentation(robotType, payload.unit, name, value, model, { [name]: [target] });
                if (mapped.modelValue !== null) model.setJointValue(target, mapped.modelValue);
            }
        }
    }, [definition.joint_map, joints, model, payload.unit, robotType]);

    // Models are mutable and cached. Restore the selected model when this live
    // view leaves, rather than leaking its measured pose into the next form.
    useEffect(
        () => () => {
            if (model !== undefined) Object.keys(model.joints).forEach((name) => model.setJointValue(name, 0));
        },
        [model]
    );

    if (readings.length === 0) return null;
    return (
        <Disclosure defaultExpanded>
            <DisclosureTitle>Joint states (read only)</DisclosureTitle>
            <DisclosurePanel>
                <Flex direction='column' gap='size-100'>
                    {readings.map((joint) => (
                        <View key={joint.name}>
                            <Text>
                                {joint.name.replace(/\.pos$/, '')}: {joint.value.toFixed(2)} {joint.unit}
                            </Text>
                            {joint.range !== null && (
                                <Slider
                                    aria-label={joint.name}
                                    isDisabled
                                    value={joint.value}
                                    minValue={joint.range[0]}
                                    maxValue={joint.range[1]}
                                />
                            )}
                            {joint.range !== null && (joint.value < joint.range[0] || joint.value > joint.range[1]) && (
                                <Text>Outside the configured display range</Text>
                            )}
                        </View>
                    ))}
                </Flex>
            </DisclosurePanel>
        </Disclosure>
    );
};

export const LiveConnectPanel = () => {
    const live = useLiveConnect();
    const { activeType } = useRobotForm();
    const [confirming, setConfirming] = useState(false);
    const status = connectionStatus(live);

    return (
        <View padding='size-200' backgroundColor='gray-100' UNSAFE_style={{ maxHeight: '45%', overflowY: 'auto' }}>
            <Flex direction='column' gap='size-150'>
                <Heading level={3}>Live connect</Heading>
                <Text>
                    Connecting or disconnecting can move the arm or change motor torque. Keep the workspace clear and
                    support the arm when needed.
                </Text>
                <div role='status'>{status}</div>
                {live.error !== null && <div role='alert'>{live.error}</div>}
                {live.validationError !== null && <div role='alert'>{live.validationError}</div>}
                <Flex gap='size-100'>
                    {live.enabled ? (
                        <Button
                            variant='secondary'
                            isDisabled={live.phase === 'releasing'}
                            onPress={() => live.setEnabled(false)}
                        >
                            Disconnect
                        </Button>
                    ) : (
                        <Button
                            variant='secondary'
                            isDisabled={activeType === undefined || live.phase === 'releasing'}
                            onPress={() => setConfirming(true)}
                        >
                            Enable Live connect
                        </Button>
                    )}
                    <DialogContainer onDismiss={() => setConfirming(false)}>
                        {confirming && (
                            <AlertDialog
                                title='Enable Live connect?'
                                variant='warning'
                                primaryActionLabel='Enable'
                                secondaryActionLabel='Cancel'
                                onPrimaryAction={() => {
                                    setConfirming(false);
                                    live.setEnabled(true);
                                }}
                                onSecondaryAction={() => setConfirming(false)}
                            >
                                The driver can move the robot on connection and disconnection, including homing the arm.
                                After enabling, valid configuration changes reconnect automatically. Make sure it is
                                safe to proceed.
                            </AlertDialog>
                        )}
                    </DialogContainer>
                    {live.enabled && live.phase === 'error' && (
                        <Button variant='secondary' onPress={live.retry}>
                            Retry
                        </Button>
                    )}
                </Flex>
                {activeType !== undefined && <LiveJointStates robotType={activeType} />}
                {live.verified && (
                    <Text>
                        Model visualization depends on the plugin&apos;s joint conventions. SO101 normalized-to-model
                        mapping is approximate; unknown plugin units are shown as raw values without model animation.
                    </Text>
                )}
            </Flex>
        </View>
    );
};

function connectionStatus(live: ReturnType<typeof useLiveConnect>): string {
    if (live.phase === 'releasing') return 'Releasing robot connection…';
    if (live.needsSetup && live.enabled) return 'Complete setup or upload calibration to enable live preview.';
    if (!live.enabled) return 'Live connect is off';
    if (live.validationError !== null) return live.validationError;
    if (live.validatedPayload === null) return 'Waiting for a valid configuration…';
    if (live.verified) return 'Connected — showing measured joint states';
    return live.phase === 'error' ? 'Connection failed' : 'Connecting…';
}
