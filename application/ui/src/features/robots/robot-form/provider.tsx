import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../robot-types';

export type RobotForm = {
    name: string;
    type: SchemaRobotType;
    connection_string: string;
    serial_number: string;
    serial_number_left: string;
    serial_number_right: string;
    active_calibration_id_left: string;
    active_calibration_id_right: string;
    connection_string_left: string;
    connection_string_right: string;
};

export type RobotFormState = RobotForm | null;

export const RobotFormContext = createContext<RobotFormState>(null);
export const SetRobotFormContext = createContext<Dispatch<SetStateAction<RobotForm>> | null>(null);

export const buildRobotBodyFromForm = (robotForm: RobotForm, robot_id: string): SchemaRobotInput | null => {
    if (!robotForm.type) {
        return null;
    }

    switch (robotForm.type) {
        case 'SO101_Follower':
        case 'SO101_Leader':
            if (!robotForm.serial_number) {
                return null;
            }

            return {
                id: robot_id,
                name: robotForm.name,
                type: robotForm.type,
                payload: {
                    connection_string: robotForm.connection_string ?? '',
                    serial_number: robotForm.serial_number,
                },
            };
        case 'SO101_Bimanual_Follower':
        case 'SO101_Bimanual_Leader':
            if (
                !robotForm.serial_number_left ||
                !robotForm.serial_number_right ||
                !robotForm.active_calibration_id_left ||
                !robotForm.active_calibration_id_right
            ) {
                return null;
            }

            return {
                id: robot_id,
                name: robotForm.name,
                type: robotForm.type,
                payload: {
                    connection_string_left: robotForm.connection_string_left ?? '',
                    connection_string_right: robotForm.connection_string_right ?? '',
                    serial_number_left: robotForm.serial_number_left,
                    serial_number_right: robotForm.serial_number_right,
                    active_calibration_id_left: robotForm.active_calibration_id_left,
                    active_calibration_id_right: robotForm.active_calibration_id_right,
                },
            };
        case 'Trossen_WidowXAI_Follower':
        case 'Trossen_WidowXAI_Leader':
            if (!robotForm.connection_string) {
                return null;
            }

            return {
                id: robot_id,
                name: robotForm.name,
                type: robotForm.type,
                payload: {
                    connection_string: robotForm.connection_string,
                    serial_number: robotForm.serial_number ?? '',
                },
            };
        case 'Trossen_Bimanual_WidowXAI_Follower':
        case 'Trossen_Bimanual_WidowXAI_Leader':
            if (!robotForm.connection_string_left || !robotForm.connection_string_right) {
                return null;
            }

            return {
                id: robot_id,
                name: robotForm.name,
                type: robotForm.type,
                payload: {
                    connection_string_left: robotForm.connection_string_left,
                    connection_string_right: robotForm.connection_string_right,
                    serial_number: robotForm.serial_number ?? '',
                },
            };
        default:
            return null;
    }
};

export const useRobotFormBody = (robot_id: string): SchemaRobotInput | null => {
    const robotForm = useRobotForm();

    if (robotForm === undefined) {
        return null;
    }

    return buildRobotBodyFromForm(robotForm, robot_id);
};

export const RobotFormProvider = ({ children, robot }: { children: ReactNode; robot?: SchemaRobot }) => {
    const initialConnectionString =
        robot !== undefined && 'connection_string' in robot.payload ? robot.payload.connection_string : '';
    const initialSerialNumber = robot?.payload.serial_number ?? '';
    const initialLeftSerial = robot !== undefined && 'serial_number_left' in robot.payload ? robot.payload.serial_number_left : '';
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
