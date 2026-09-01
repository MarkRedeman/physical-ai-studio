export type CameraFingerprint = Record<string, unknown>;

export const fingerprintKey = (fingerprint: CameraFingerprint | null | undefined): string | undefined => {
    if (!fingerprint) return undefined;
    const canonicalize = (value: unknown): unknown => {
        if (Array.isArray(value)) return value.map(canonicalize);
        if (value !== null && typeof value === 'object') {
            return Object.fromEntries(
                Object.entries(value)
                    .map(([key, item]): [string, unknown] => [key, canonicalize(item)])
                    .sort(([a], [b]) => a.localeCompare(b))
            );
        }
        return value;
    };

    return JSON.stringify(canonicalize(fingerprint));
};

export const formatFingerprint = (fingerprint: CameraFingerprint | null | undefined): string => {
    if (!fingerprint) return 'Camera needs reselection';
    const serial = fingerprint.serial;
    if (typeof serial === 'string' && serial) return serial;

    const url = fingerprint.url;
    if (typeof url === 'string' && url) return url;

    const bus = fingerprint.bus;
    const index = fingerprint.index;
    if (typeof bus === 'string' && bus.includes('v4l2loopback')) {
        return typeof index === 'number' ? `Virtual camera ${index}` : 'Virtual camera';
    }
    if (typeof bus === 'string' && bus) return bus;
    if (typeof index === 'number') return `Camera ${index}`;

    return fingerprintKey(fingerprint) ?? 'Camera needs reselection';
};
