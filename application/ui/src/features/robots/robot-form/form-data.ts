import {
    SchemaReBotB601DMRobotInput,
    SchemaReBotArm102LeaderRobotInput,
    SchemaRobotInput,
    SchemaRobotType,
} from '../robot-types';

export type RobotFormFields = {
    name: string;
    type: SchemaRobotType;
    connection_string: string;
    serial_number: string;
    connection_string_left: string;
    connection_string_right: string;
    can_adapter: 'damiao' | 'socketcan';
    dm_serial_baud: string;
    disable_torque_on_disconnect: boolean;
    force_pos_torque_ratio: string;
    baudrate: string;
    unlock_on_connect: boolean;
    reset_multi_turn_on_connect: boolean;
    zero_on_connect: boolean;
};

const toStringValue = (input: FormDataEntryValue | null): string => {
    if (input === null) {
        return '';
    }
    return typeof input === 'string' ? input : input.name;
};

const toBooleanValue = (input: FormDataEntryValue | null, fallback: boolean): boolean => {
    if (input === null) {
        return fallback;
    }
    return toStringValue(input) === 'true';
};

const toCanAdapter = (
    input: FormDataEntryValue | null,
    fallback: RobotFormFields['can_adapter']
): RobotFormFields['can_adapter'] => {
    const value = toStringValue(input);
    if (value === 'socketcan') {
        return 'socketcan';
    }
    if (value === 'damiao') {
        return 'damiao';
    }
    return fallback;
};

export const parseRobotFormFromFormData = (
    formData: FormData,
    fallback: Pick<
        RobotFormFields,
        | 'type'
        | 'connection_string'
        | 'serial_number'
        | 'connection_string_left'
        | 'connection_string_right'
        | 'can_adapter'
        | 'dm_serial_baud'
        | 'disable_torque_on_disconnect'
        | 'force_pos_torque_ratio'
        | 'baudrate'
        | 'unlock_on_connect'
        | 'reset_multi_turn_on_connect'
        | 'zero_on_connect'
    >
): RobotFormFields => {
    return {
        name: toStringValue(formData.get('name')),
        type: fallback.type,
        connection_string: toStringValue(formData.get('payload.connection_string')) || fallback.connection_string,
        serial_number: toStringValue(formData.get('payload.serial_number')) || fallback.serial_number,
        connection_string_left:
            toStringValue(formData.get('payload.connection_string_left')) || fallback.connection_string_left,
        connection_string_right:
            toStringValue(formData.get('payload.connection_string_right')) || fallback.connection_string_right,
        can_adapter: toCanAdapter(formData.get('payload.can_adapter'), fallback.can_adapter),
        dm_serial_baud: toStringValue(formData.get('payload.dm_serial_baud')) || fallback.dm_serial_baud,
        disable_torque_on_disconnect: toBooleanValue(
            formData.get('payload.disable_torque_on_disconnect'),
            fallback.disable_torque_on_disconnect
        ),
        force_pos_torque_ratio:
            toStringValue(formData.get('payload.force_pos_torque_ratio')) || fallback.force_pos_torque_ratio,
        baudrate: toStringValue(formData.get('payload.baudrate')) || fallback.baudrate,
        unlock_on_connect: toBooleanValue(formData.get('payload.unlock_on_connect'), fallback.unlock_on_connect),
        reset_multi_turn_on_connect: toBooleanValue(
            formData.get('payload.reset_multi_turn_on_connect'),
            fallback.reset_multi_turn_on_connect
        ),
        zero_on_connect: toBooleanValue(formData.get('payload.zero_on_connect'), fallback.zero_on_connect),
    };
};

export const buildRobotBodyFromFields = (robotForm: RobotFormFields, robot_id: string): SchemaRobotInput | null => {
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
        case 'ReBot_B601_DM_Follower':
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
                    can_adapter: robotForm.can_adapter,
                    dm_serial_baud: Number(robotForm.dm_serial_baud),
                    disable_torque_on_disconnect: robotForm.disable_torque_on_disconnect,
                    force_pos_torque_ratio: Number(robotForm.force_pos_torque_ratio),
                },
            };
        case 'ReBot_Arm102_Leader':
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
                    baudrate: Number(robotForm.baudrate),
                    unlock_on_connect: robotForm.unlock_on_connect,
                    reset_multi_turn_on_connect: robotForm.reset_multi_turn_on_connect,
                    zero_on_connect: robotForm.zero_on_connect,
                },
            };
        default:
            return null;
    }
};
