import { useCallback, useEffect, useState } from 'react';

import { sortBy } from 'lodash-es';
import useWebSocket from 'react-use-websocket';
import { degToRad } from 'three/src/math/MathUtils.js';

import { fetchClient } from '../../api/client';
import { useRobotModels } from './robot-models-context';

type JointsState = Array<{
    name: string;
    value: number;
}>;

type StateWasUpdatedEvent = {
    name: 'state_was_updated';
    is_controlled: boolean;
    // [joint_name]: robot state in degrees
    state: Record<string, number>;
};

const getNewJointState = (newJoints: StateWasUpdatedEvent['state']) => {
    return sortBy(
        Object.keys(newJoints).map((joint_name) => {
            return {
                name: joint_name,
                value: Number(newJoints[joint_name]),
            };
        }),
        (joint) => joint.name
    );
};

export const useSynchronizeModelJoints = (joints: JointsState, urdfPath: string) => {
    const { getModel } = useRobotModels();
    const model = getModel(urdfPath);

    function removeSuffix(str: string, suffix: string): string {
        return str.endsWith(suffix) ? str.slice(0, -suffix.length) : str;
    }

    useEffect(() => {
        if (!model) return;

        joints.forEach((joint) => {
            const name = removeSuffix(joint.name, '.pos');

            if (name === 'gripper' && model.robotName == 'wxai') {
                model.setJointValue('left_carriage_joint', joint.value); // meters
                return;
            }

            if (model.joints[name]) {
                model.setJointValue(name, degToRad(joint.value));
            }
        });
    }, [model, joints]);
};

export const useJointState = (project_id: string, robot_id: string) => {
    const [joints, setJoints] = useState<JointsState>([]);

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const payload = JSON.parse(event.data);

            if (payload['event'] === 'state_was_updated') {
                const newJoints = getNewJointState(payload['state']);
                setJoints(newJoints);
            }
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }, []);

    const serial_number = '5AAF270363';
    const url = `http://localhost:8000/api/robots/${serial_number}/ws`;
    const queryParams = {
        fps: 120,
        driver: 'feetech',
        normalize: true,
        calibration_id: 'leader',
    };

    const socket = useWebSocket(url, {
        queryParams: queryParams,
        share: true,
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });
    // const socket = useWebSocket(`/api/projects/${project_id}/robots/${robot_id}/ws`, {
    //     queryParams: {
    //         fps: 30,
    //     },
    //     share: true,
    //     shouldReconnect: () => true,
    //     reconnectAttempts: 5,
    //     reconnectInterval: 3000,
    //     onMessage: handleMessage,
    //     onError: (error) => console.error('WebSocket error:', error),
    //     onClose: () => console.info('WebSocket closed'),
    // });

    return { joints, socket };
};
