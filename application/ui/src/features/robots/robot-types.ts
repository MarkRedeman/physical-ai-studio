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

type SchemaSo101BimanualType = 'SO101_Bimanual_Follower' | 'SO101_Bimanual_Leader';

type SchemaSo101BimanualPayload = {
    connection_string_left: string;
    connection_string_right: string;
    serial_number_left: string;
    serial_number_right: string;
    active_calibration_id_left?: string | null;
    active_calibration_id_right?: string | null;
};

export type SchemaSo101BimanualRobotInput = {
    id: string;
    name: string;
    type: SchemaSo101BimanualType;
    payload: SchemaSo101BimanualPayload;
    active_calibration_id?: string | null;
};

export type SchemaSo101BimanualRobotOutput = SchemaSo101BimanualRobotInput & {
    created_at?: string | null;
    updated_at?: string | null;
};

export type SchemaSo101BimanualRobotWithConnectionState = SchemaSo101BimanualRobotOutput & {
    connection_status: 'online' | 'offline' | 'unknown';
};

/** Union of all concrete robot output schemas (as returned by the API). */
export type SchemaRobot =
    | SchemaSo101RobotOutput
    | SchemaSo101BimanualRobotOutput
    | SchemaTrossenSingleArmRobotOutput
    | SchemaTrossenBimanualRobotOutput;

/** Union of all concrete robot input schemas (for create/update requests). */
export type SchemaRobotInput =
    | SchemaSo101RobotInput
    | SchemaSo101BimanualRobotInput
    | SchemaTrossenSingleArmRobotInput
    | SchemaTrossenBimanualRobotInput;

/** All possible robot type discriminators. */
export type SchemaRobotType = SchemaRobot['type'];

/** Union of all robot-with-connection-state schemas (as returned by the online endpoint). */
export type SchemaRobotWithConnectionState =
    | SchemaSo101RobotWithConnectionState
    | SchemaSo101BimanualRobotWithConnectionState
    | SchemaTrossenSingleArmRobotWithConnectionState
    | SchemaTrossenBimanualRobotWithConnectionState;
