import type { SchemaExportVariantDetail } from '../../api/openapi-spec';

export const getMockedExportVariant = (
    overrides: Partial<SchemaExportVariantDetail> = {}
): SchemaExportVariantDetail => ({
    schema_version: 1,
    id: 'original',
    source_variant: 'original',
    checkpoint_kind: 'ordinary',
    rtc_enabled: false,
    weight_compression: null,
    label: 'Original',
    size_bytes: 1048576,
    file_count: 2,
    can_run_in_studio: true,
    available_operations: [{ operation: 'int8_sym', target_variant: 'int8', available: true }],
    ...overrides,
});
