import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../robot-types';
import { buildRobotBodyFromFields, parseRobotFormFromFormData, RobotFormFields } from './form-data';

export type RobotForm = RobotFormFields;

export type RobotFormState = RobotForm | null;

export const RobotFormContext = createContext<RobotFormState>(null);
export const SetRobotFormContext = createContext<Dispatch<SetStateAction<RobotForm>> | null>(null);

export const buildRobotBodyFromForm = (robotForm: RobotForm, robot_id: string): SchemaRobotInput | null => {
    return buildRobotBodyFromFields(robotForm, robot_id);
};

export const buildRobotBodyFromFormElement = (
    formElement: HTMLFormElement,
    fallback: RobotForm,
    robot_id: string
): SchemaRobotInput | null => {
    const formData = new FormData(formElement);
    const parsed = parseRobotFormFromFormData(formData, fallback);
    return buildRobotBodyFromFields(parsed, robot_id);
};

export const useRobotFormBodyFromElement = (robot_id: string, formElement: HTMLFormElement | null): SchemaRobotInput | null => {
    const robotForm = useRobotForm();

    if (robotForm === undefined || formElement === null) {
        return null;
    }

    return buildRobotBodyFromFormElement(formElement, robotForm, robot_id);
};

export const RobotFormProvider = ({ children, robot }: { children: ReactNode; robot?: SchemaRobot }) => {
    const initialConnectionString =
        robot !== undefined && 'connection_string' in robot.payload ? robot.payload.connection_string : '';
    const initialSerialNumber = robot !== undefined && 'serial_number' in robot.payload ? robot.payload.serial_number : '';
    const initialLeftSerial =
        robot !== undefined && 'serial_number_left' in robot.payload ? robot.payload.serial_number_left : '';
    const initialRightSerial =
        robot !== undefined && 'serial_number_right' in robot.payload ? robot.payload.serial_number_right : '';
    const initialLeftCalibration =
        robot !== undefined && 'active_calibration_id_left' in robot.payload
            ? (robot.payload.active_calibration_id_left ?? '')
            : '';
    const initialRightCalibration =
        robot !== undefined && 'active_calibration_id_right' in robot.payload
            ? (robot.payload.active_calibration_id_right ?? '')
            : '';
    const initialLeftConnection =
        robot !== undefined && 'connection_string_left' in robot.payload ? robot.payload.connection_string_left : '';
    const initialRightConnection =
        robot !== undefined && 'connection_string_right' in robot.payload ? robot.payload.connection_string_right : '';
    const initialCanAdapter =
        robot !== undefined && 'can_adapter' in robot.payload && robot.payload.can_adapter === 'socketcan'
            ? 'socketcan'
            : 'damiao';
    const initialDmSerialBaud =
        robot !== undefined && 'dm_serial_baud' in robot.payload ? String(robot.payload.dm_serial_baud) : '921600';
    const initialDisableTorqueOnDisconnect =
        robot !== undefined && 'disable_torque_on_disconnect' in robot.payload
            ? robot.payload.disable_torque_on_disconnect
            : true;
    const initialForcePosTorqueRatio =
        robot !== undefined && 'force_pos_torque_ratio' in robot.payload
            ? String(robot.payload.force_pos_torque_ratio)
            : '0.1';
    const initialBaudrate = robot !== undefined && 'baudrate' in robot.payload ? String(robot.payload.baudrate) : '1000000';
    const initialUnlockOnConnect =
        robot !== undefined && 'unlock_on_connect' in robot.payload ? robot.payload.unlock_on_connect : true;
    const initialResetMultiTurnOnConnect =
        robot !== undefined && 'reset_multi_turn_on_connect' in robot.payload
            ? robot.payload.reset_multi_turn_on_connect
            : true;
    const initialZeroOnConnect = robot !== undefined && 'zero_on_connect' in robot.payload ? robot.payload.zero_on_connect : false;

    const [value, setValue] = useState<RobotForm>({
        name: robot?.name ?? '',
        type: robot?.type ?? 'SO101_Follower',
        connection_string: initialConnectionString,
        serial_number: initialSerialNumber,
        serial_number_left: initialLeftSerial,
        serial_number_right: initialRightSerial,
        active_calibration_id_left: initialLeftCalibration,
        active_calibration_id_right: initialRightCalibration,
        connection_string_left: initialLeftConnection,
        connection_string_right: initialRightConnection,
        can_adapter: initialCanAdapter,
        dm_serial_baud: initialDmSerialBaud,
        disable_torque_on_disconnect: initialDisableTorqueOnDisconnect,
        force_pos_torque_ratio: initialForcePosTorqueRatio,
        baudrate: initialBaudrate,
        unlock_on_connect: initialUnlockOnConnect,
        reset_multi_turn_on_connect: initialResetMultiTurnOnConnect,
        zero_on_connect: initialZeroOnConnect,
    });

    return (
        <RobotFormContext.Provider value={value}>
            <SetRobotFormContext.Provider value={setValue}>{children}</SetRobotFormContext.Provider>
        </RobotFormContext.Provider>
    );
};

export const useRobotForm = () => {
    const context = useContext(RobotFormContext);

    if (context === null) {
        throw new Error('useRobotForm was used outside of RobotFormProvider');
    }

    return context;
};

export const useSetRobotForm = () => {
    const context = useContext(SetRobotFormContext);

    if (context === null) {
        throw new Error('useSetRobotForm was used outside of RobotFormProvider');
    }

    return context;
};
