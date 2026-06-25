import { SchemaRobotInput, SchemaRobotType } from '../robot-types';
import { buildSO101Body, type SO101FormData } from './catalog/so101';
import { buildWidowxBody, type WidowxFormData } from './catalog/widowxai';
import { buildBimanualBody, type BimanualFormData } from './catalog/widowxai-bimanual';

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
        case 'so101':
            return buildSO101Body(formData as SO101FormData, schemaType, robot_id);
        case 'widowx':
            return buildWidowxBody(formData as WidowxFormData, schemaType, robot_id);
        case 'bimanual_widowx':
            return buildBimanualBody(formData as BimanualFormData, schemaType, robot_id);
    }
};
