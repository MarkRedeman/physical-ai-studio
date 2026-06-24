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
