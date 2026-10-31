import { SchemaRobotCatalogDefinitionResponse } from '../../api/openapi-spec';

export const getMockedRobotCatalogDefinition = (
    overrides: Partial<SchemaRobotCatalogDefinitionResponse> = {}
): SchemaRobotCatalogDefinitionResponse => ({
    type: 'Trossen_WidowXAI_Follower',
    display_name: 'WidowX AI',
    role: 'follower',
    urdf_path: null,
    package_map: {},
    joint_map: {},
    zero_calibration: null,
    ...overrides,
});
