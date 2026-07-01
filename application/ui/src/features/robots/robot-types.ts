import {
    SchemaReBotArm102LeaderRobotInput as SchemaReBotArm102LeaderRobotInputSpec,
    SchemaReBotArm102LeaderRobotOutput as SchemaReBotArm102LeaderRobotOutputSpec,
    SchemaReBotArm102LeaderRobotWithConnectionState as SchemaReBotArm102LeaderRobotWithConnectionStateSpec,
    SchemaReBotB601DmPayload,
    SchemaReBotB601DmRobotInput,
    SchemaReBotB601DmRobotOutput,
    SchemaReBotB601DmRobotWithConnectionState,
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

export type SchemaReBotB601DMRobotInput = SchemaReBotB601DmRobotInput;
export type SchemaReBotB601DMRobotOutput = SchemaReBotB601DmRobotOutput;
export type SchemaReBotB601DMRobotWithConnectionState = SchemaReBotB601DmRobotWithConnectionState;
export type SchemaReBotB601DMPayload = SchemaReBotB601DmPayload;

export type SchemaReBotArm102LeaderRobotInput = SchemaReBotArm102LeaderRobotInputSpec;
export type SchemaReBotArm102LeaderRobotOutput = SchemaReBotArm102LeaderRobotOutputSpec;
export type SchemaReBotArm102LeaderRobotWithConnectionState = SchemaReBotArm102LeaderRobotWithConnectionStateSpec;

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
