import { useEffect, useState } from 'react';

import { Button, Flex, Heading, Text } from '@geti/ui';
import { useNavigate } from 'react-router';
import { degToRad } from 'three/src/math/MathUtils.js';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../../api/client';
import { SchemaRobot } from '../../../../api/openapi-spec';
import { paths } from '../../../../router';
import { useProjectId } from '../../../projects/use-project';
import { useRobotForm } from '../../robot-form/provider';
import { urdfPathForType, useRobotModels } from '../../robot-models-context';
import { useTrossenSetupActions, useTrossenSetupState } from './wizard-provider';

import classes from '../shared/setup-wizard.module.scss';

// ---------------------------------------------------------------------------
// Hook: sync joint state from the setup websocket to the URDF model
// ---------------------------------------------------------------------------

/**
 * Maps backend motor names to wxai URDF joint names.
 *
 * The backend sends positions keyed by semantic motor name
 * (e.g. `shoulder_pan.pos`, `shoulder_lift.pos`) but the wxai URDF uses
 * positional names (`joint_0` … `joint_5`) for the six revolute joints
 * and `left_carriage_joint` for the prismatic gripper.
 */
const MOTOR_TO_URDF_JOINT: Record<string, string> = {
    shoulder_pan: 'joint_0',
    shoulder_lift: 'joint_1',
    elbow_flex: 'joint_2',
    wrist_flex: 'joint_3',
    wrist_yaw: 'joint_4',
    wrist_roll: 'joint_5',
    gripper: 'gripper',
};

/**
 * Syncs joint positions (from `state_was_updated` events) to the loaded URDF
 * model. Trossen positions are in degrees (gripper in meters) — same format
 * as the Trossen robot client's `read_state()`.
 *
 * For the wxai URDF, the gripper joint is mapped to `left_carriage_joint`
 * (the URDF uses a prismatic joint in meters, matching the raw gripper value).
 */
const useSyncJointState = (jointState: Record<string, number> | null, robotType: string) => {
    const { getModel } = useRobotModels();
    const urdfPath = urdfPathForType(robotType as Parameters<typeof urdfPathForType>[0]);
    const model = getModel(urdfPath);

    useEffect(() => {
        if (!jointState || !model) {
            return;
        }

        for (const [key, value] of Object.entries(jointState)) {
            const name = key.endsWith('.pos') ? key.slice(0, -4) : key;

            console.log(jointState, name, model.joints);

            // wxai URDF maps the gripper to a prismatic joint (meters, no conversion)
            if (name === 'gripper' && model.robotName === 'wxai') {
                model.setJointValue('left_carriage_joint', value);
                continue;
            }

            if (model.joints[MOTOR_TO_URDF_JOINT[name]]) {
                model.setJointValue(MOTOR_TO_URDF_JOINT[name], degToRad(value));
            }
        }
    }, [model, jointState]);
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Trossen verification step — the backend broadcast loop auto-streams live
 * joint positions during the VERIFICATION phase and syncs them to the 3D
 * URDF model so the user can visually verify the robot moves correctly.
 * Shows "Save Robot" button that persists the robot to the DB.
 *
 * Unlike SO101, Trossen has no calibration to save — just the robot itself.
 * `is_calibrated` is hardcoded to `true` by the Trossen SDK.
 */
export const TrossenVerificationStep = () => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();
    const { goBack } = useTrossenSetupActions();
    const { wsState } = useTrossenSetupState();
    const robotForm = useRobotForm();

    const connectionString = robotForm.connection_string ?? '';
    const robotType = robotForm.type ?? '';

    const [robotId] = useState(() => uuidv4());
    const [saving, setSaving] = useState(false);
    const [saveError, setSaveError] = useState<string | null>(null);

    // Sync live joint positions to the 3D model
    useSyncJointState(wsState.jointState, robotType);

    const addRobotMutation = $api.useMutation('post', '/api/projects/{project_id}/robots');

    const robotBody: SchemaRobot | null =
        robotForm.type !== null && robotForm.name
            ? {
                  id: robotId,
                  name: robotForm.name,
                  type: robotForm.type,
                  connection_string: connectionString,
                  serial_number: robotForm.serial_number ?? '',
                  active_calibration_id: null,
              }
            : null;

    const handleSave = async () => {
        if (robotBody === null) {
            return;
        }

        setSaving(true);
        setSaveError(null);

        try {
            // Create the robot (no calibration save for Trossen)
            const createdRobot = await addRobotMutation.mutateAsync({
                params: { path: { project_id } },
                body: robotBody,
            });

            // Navigate to the robot page
            navigate(paths.project.robots.show({ project_id, robot_id: createdRobot.id }));
        } catch (err) {
            setSaveError(err instanceof Error ? err.message : 'Failed to save robot');
        } finally {
            setSaving(false);
        }
    };

    return (
        <Flex direction='column' gap='size-300'>
            <div className={classes.successBox}>
                <Text>
                    Robot is connected and ready. Move the robot arm to verify that the 3D visualization matches the
                    physical robot, then save.
                </Text>
            </div>

            {!wsState.isConnected && (
                <div className={classes.warningBox}>
                    <Text>WebSocket disconnected — 3D preview is not updating.</Text>
                </div>
            )}

            <div className={classes.sectionCard}>
                <Flex direction='column' gap='size-100'>
                    <Heading level={4}>Robot Details</Heading>
                    <Flex direction='column' gap='size-50'>
                        <Text>
                            <strong>Name:</strong> {robotForm.name}
                        </Text>
                        <Text>
                            <strong>Type:</strong> {robotType}
                        </Text>
                        <Text>
                            <strong>IP Address:</strong> {connectionString}
                        </Text>
                    </Flex>
                </Flex>
            </div>

            {(wsState.error || saveError) && (
                <div className={classes.warningBox}>
                    <Text>{saveError ?? wsState.error}</Text>
                </div>
            )}

            <Flex gap='size-200' justifyContent='space-between'>
                <Button variant='secondary' onPress={goBack}>
                    Back
                </Button>
                <Button variant='accent' isPending={saving} isDisabled={robotBody === null} onPress={handleSave}>
                    Save Robot
                </Button>
            </Flex>
        </Flex>
    );
};
