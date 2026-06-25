import { SchemaRobotInput, SchemaRobotType } from '../robot-types';
import type { SO101FormData } from './catalog/so101';
import type { WidowxFormData } from './catalog/widowxai';
import type { BimanualFormData } from './catalog/widowxai-bimanual';

export type RobotType = 'so101' | 'widowx' | 'bimanual_widowx';

export type AnyRobotFormData = SO101FormData | WidowxFormData | BimanualFormData;

export type RobotTypeData = {
    so101: SO101FormData;
    widowx: WidowxFormData;
    bimanual_widowx: BimanualFormData;
};

export const typeForSchema: Record<SchemaRobotType, RobotType> = {
    SO101_Follower: 'so101',
    SO101_Leader: 'so101',
    Trossen_WidowXAI_Follower: 'widowx',
    Trossen_WidowXAI_Leader: 'widowx',
    Trossen_Bimanual_WidowXAI_Follower: 'bimanual_widowx',
    Trossen_Bimanual_WidowXAI_Leader: 'bimanual_widowx',
};

export const buildRobotBody = (
    robotType: RobotType,
    formData: AnyRobotFormData,
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    switch (robotType) {
        case 'so101': {
            const data = formData as SO101FormData;
            if (!data.serial_number) {
                return null;
            }

            return {
                id: robot_id,
                name: data.name,
                type: schemaType,
                payload: {
                    connection_string: data.connection_string ?? '',
                    serial_number: data.serial_number,
                },
            };
        }
        case 'widowx': {
            const data = formData as WidowxFormData;
            if (!data.connection_string) {
                return null;
            }

            return {
                id: robot_id,
                name: data.name,
                type: schemaType,
                payload: {
                    connection_string: data.connection_string,
                    serial_number: data.serial_number ?? '',
                },
            };
        }
        case 'bimanual_widowx': {
            const data = formData as BimanualFormData;
            if (!data.connection_string_left || !data.connection_string_right) {
                return null;
            }

            return {
                id: robot_id,
                name: data.name,
                type: schemaType,
                payload: {
                    connection_string_left: data.connection_string_left,
                    connection_string_right: data.connection_string_right,
                    serial_number: data.serial_number ?? '',
                },
            };
        }
    }
};
