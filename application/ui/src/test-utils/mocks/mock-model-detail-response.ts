import { SchemaModelDetailResponse } from '../../api/openapi-spec';
import { getMockedExportVariant } from './mock-export-variant';
import { getMockedModel } from './mock-model';

export const getMockedModelDetailResponse = (
    overrides: Partial<SchemaModelDetailResponse> = {}
): SchemaModelDetailResponse => ({
    model: getMockedModel(),
    exports: [{ type: 'openvino', size_bytes: 1048576, file_count: 2, variants: [getMockedExportVariant()] }],
    training_summary: null,
    hparams: null,
    ...overrides,
});
