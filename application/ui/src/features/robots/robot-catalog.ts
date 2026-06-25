import { SchemaRobotCatalogEntry, SchemaRobotType } from '../../api/openapi-spec';
import { $api } from '../../api/client';
import { SchemaRobot } from './robot-types';

export const useRobotCatalogQuery = () => {
    return $api.useSuspenseQuery('get', '/api/robots/catalog', {
        meta: { skipInvalidation: true },
    });
};

export const useRobotCatalogMap = () => {
    const query = useRobotCatalogQuery();
    const byType = new Map<SchemaRobotType, SchemaRobotCatalogEntry>();

    query.data.forEach((entry) => {
        byType.set(entry.type, entry);
    });

    return {
        entries: query.data,
        byType,
    };
};

export const useRobotRoleChecks = () => {
    const { byType } = useRobotCatalogMap();

    const isFollower = (robot: Pick<SchemaRobot, 'type'>) => {
        const entry = byType.get(robot.type);
        if (entry === undefined) {
            throw new Error(`Missing catalog entry for robot type: ${robot.type}`);
        }
        return entry.role === 'follower';
    };

    const isLeader = (robot: Pick<SchemaRobot, 'type'>) => {
        const entry = byType.get(robot.type);
        if (entry === undefined) {
            throw new Error(`Missing catalog entry for robot type: ${robot.type}`);
        }
        return entry.role === 'leader';
    };

    return {
        isFollower,
        isLeader,
    };
};

/**
 * Resolve the URL of the root URDF file for a robot type.
 *
 * This value is passed to `URDFLoader.load(...)` as the initial document path.
 *
 * Example: `/api/robots/catalog/ReBot_Arm102_Leader/urdf`
 */
export const useUrdfPathForType = () => {
    const { byType } = useRobotCatalogMap();

    return (robotType: SchemaRobotType): string => {
        const entry = byType.get(robotType);
        if (entry === undefined) {
            throw new Error(`Missing catalog entry for robot type: ${robotType}`);
        }
        if (!entry.urdf_path) {
            throw new Error(`Missing catalog URDF path for robot type: ${robotType}`);
        }
        return entry.urdf_path;
    };
};

/**
 * Resolve observation-to-URDF joint mapping for a robot type.
 *
 * The returned map translates Studio joint feature names (e.g. `shoulder_pan.pos`)
 * to URDF joint names used by the loaded model.
 */
export const useJointMapForType = () => {
    const { byType } = useRobotCatalogMap();

    return (robotType: SchemaRobotType): Record<string, string[]> => {
        const entry = byType.get(robotType);
        if (entry === undefined) {
            throw new Error(`Missing catalog entry for robot type: ${robotType}`);
        }
        if (!entry.joint_map) {
            throw new Error(`Missing catalog joint map for robot type: ${robotType}`);
        }
        return entry.joint_map as Record<string, string[]>;
    };
};

/**
 * Resolve URDF package mappings for a robot type.
 *
 * URDF files can reference meshes/textures via `package://<name>/...` paths.
 * `URDFLoader` needs a package map (`<name> -> URL prefix`) to fetch those assets.
 *
 * Example: `stararm102 -> /api/robots/catalog/ReBot_Arm102_Leader`
 */
export const usePackageMapForType = () => {
    const { byType } = useRobotCatalogMap();

    return (robotType: SchemaRobotType): Record<string, string> => {
        const entry = byType.get(robotType);
        if (entry === undefined) {
            throw new Error(`Missing catalog entry for robot type: ${robotType}`);
        }
        return (entry.package_map ?? {}) as Record<string, string>;
    };
};
