import {
    SchemaSo101RobotInput,
    SchemaSo101RobotOutput,
    SchemaSo101RobotWithConnectionState,
    SchemaTrossenBimanualRobotInput,
    SchemaTrossenBimanualRobotOutput,
    SchemaTrossenBimanualRobotWithConnectionState,
    SchemaTrossenSingleArmRobotInput,
    SchemaTrossenSingleArmRobotOutput,
    SchemaTrossenSingleArmRobotWithConnectionState,
} from '../../api/openapi-spec';

type SchemaReBotB601DMType = 'ReBot_B601_DM_Follower';
type SchemaReBotArm102LeaderType = 'ReBot_Arm102_Leader';

type SchemaReBotB601DMPayload = {
    connection_string: string;
    serial_number: string;
    can_adapter: 'damiao' | 'socketcan';
    dm_serial_baud: number;
    disable_torque_on_disconnect: boolean;
    force_pos_torque_ratio: number;
};

type SchemaReBotArm102LeaderPayload = {
    connection_string: string;
    serial_number: string;
    baudrate: number;
    unlock_on_connect: boolean;
    reset_multi_turn_on_connect: boolean;
    zero_on_connect: boolean;
};

export type SchemaReBotB601DMRobotInput = {
    id: string;
    name: string;
    type: SchemaReBotB601DMType;
    payload: SchemaReBotB601DMPayload;
    active_calibration_id?: string | null;
};

export type SchemaReBotB601DMRobotOutput = SchemaReBotB601DMRobotInput & {
    created_at?: string | null;
    updated_at?: string | null;
};

export type SchemaReBotB601DMRobotWithConnectionState = SchemaReBotB601DMRobotOutput & {
    connection_status: 'online' | 'offline' | 'unknown';
};

export type SchemaReBotArm102LeaderRobotInput = {
    id: string;
    name: string;
    type: SchemaReBotArm102LeaderType;
    payload: SchemaReBotArm102LeaderPayload;
    active_calibration_id?: string | null;
};

export type SchemaReBotArm102LeaderRobotOutput = SchemaReBotArm102LeaderRobotInput & {
    created_at?: string | null;
    updated_at?: string | null;
};

export type SchemaReBotArm102LeaderRobotWithConnectionState = SchemaReBotArm102LeaderRobotOutput & {
    connection_status: 'online' | 'offline' | 'unknown';
};

/** Union of all concrete robot output schemas (as returned by the API). */
export type SchemaRobot =
    | SchemaSo101RobotOutput
    | SchemaReBotB601DMRobotOutput
    | SchemaReBotArm102LeaderRobotOutput
    | SchemaTrossenSingleArmRobotOutput
    | SchemaTrossenBimanualRobotOutput;

/** Union of all concrete robot input schemas (for create/update requests). */
export type SchemaRobotInput =
    | SchemaSo101RobotInput
    | SchemaReBotB601DMRobotInput
    | SchemaReBotArm102LeaderRobotInput
    | SchemaTrossenSingleArmRobotInput
    | SchemaTrossenBimanualRobotInput;

/** All possible robot type discriminators. */
export type SchemaRobotType = SchemaRobot['type'];

/** Union of all robot-with-connection-state schemas (as returned by the online endpoint). */
export type SchemaRobotWithConnectionState =
    | SchemaSo101RobotWithConnectionState
    | SchemaReBotB601DMRobotWithConnectionState
    | SchemaReBotArm102LeaderRobotWithConnectionState
    | SchemaTrossenSingleArmRobotWithConnectionState
    | SchemaTrossenBimanualRobotWithConnectionState;
