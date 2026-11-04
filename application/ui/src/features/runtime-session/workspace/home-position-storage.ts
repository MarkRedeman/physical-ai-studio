import { SchemaHomeCapability, SchemaHomePose, SchemaSessionEnvironment } from '../../../api/openapi-spec';

const isRecord = (value: unknown): value is Record<string, unknown> =>
    value !== null && typeof value === 'object' && !Array.isArray(value);

export const homeStorageKey = (project_id: string, environment: SchemaSessionEnvironment): string | null =>
    environment.environment_id == null
        ? null
        : `physicalai-studio:home:v1:${project_id}:${environment.environment_id}:${environment.robot_id}`;

/** Browser storage is untrusted: never substitute zeros, ignore unknown joints, or coerce coordinates. */
export const validateHomePose = (value: unknown, capability: SchemaHomeCapability): SchemaHomePose | null => {
    if (
        !isRecord(value) ||
        Object.keys(value).length !== 6 ||
        value.version !== 1 ||
        value.robot_id !== capability.robot_id ||
        value.fingerprint !== capability.fingerprint ||
        value.unit !== capability.unit ||
        typeof value.captured_at !== 'string' ||
        !Number.isFinite(Date.parse(value.captured_at)) ||
        !isRecord(value.positions)
    )
        return null;
    const names = Object.keys(capability.joints);
    if (names.length === 0 || names.length > 32 || Object.keys(value.positions).length !== names.length) return null;
    const positions: Record<string, number> = {};
    for (const name of names) {
        const coordinate = value.positions[name];
        const limit = capability.joints[name];
        if (
            typeof coordinate !== 'number' ||
            !Number.isFinite(coordinate) ||
            coordinate < limit.minimum ||
            coordinate > limit.maximum
        )
            return null;
        if (
            capability.unit === 'normalized' &&
            (coordinate <= limit.minimum + limit.tolerance || coordinate >= limit.maximum - limit.tolerance)
        )
            return null;
        positions[name] = coordinate;
    }
    return {
        version: 1,
        robot_id: capability.robot_id,
        fingerprint: capability.fingerprint,
        unit: capability.unit,
        captured_at: value.captured_at,
        positions,
    };
};

export const decodeStoredHome = (
    serialized: string | null,
    capability: SchemaHomeCapability
): SchemaHomePose | null => {
    if (serialized == null || serialized.length > 16_384) return null;
    try {
        return validateHomePose(JSON.parse(serialized), capability);
    } catch {
        return null;
    }
};
