import { useState } from 'react';

import { Item, Picker } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaExportVariantDetail, SchemaModelDetailResponse } from '../../../api/openapi-spec';

export type ModelFormatId = SchemaExportVariantDetail['id'];

export const useModelFormatSelection = (
    modelId: string | undefined,
    backend: string,
    initialFormatId: ModelFormatId = 'original',
    detail?: SchemaModelDetailResponse
) => {
    const query = $api.useQuery(
        'get',
        '/api/models/{model_id}',
        { params: { path: { model_id: modelId ?? '' } } },
        { enabled: modelId !== undefined && detail === undefined }
    );
    const key = `${modelId}/${backend}`;
    const [selection, setSelection] = useState<{ key: string; formatId: ModelFormatId }>();
    const formats = (detail ?? query.data)?.exports.find((item) => item.type === backend)?.variants ?? [];
    const requested = selection?.key === key ? selection.formatId : initialFormatId;
    const selected = formats.find((format) => format.id === requested) ?? formats[0];
    return {
        formats,
        selected,
        formatId: selected?.id ?? 'original',
        isPending: detail === undefined && query.isPending,
        isError: detail === undefined && query.isError,
        setFormatId: (formatId: ModelFormatId) => setSelection({ key, formatId }),
    };
};

export const ModelFormatSelection = ({
    formats,
    formatId,
    onChange,
}: {
    formats: SchemaExportVariantDetail[];
    formatId: ModelFormatId;
    onChange: (formatId: ModelFormatId) => void;
}) => (
    <Picker
        label='Model format'
        items={formats}
        selectedKey={formatId}
        onSelectionChange={(key) => {
            if (key === 'original' || key === 'int8') onChange(key);
        }}
    >
        {(item) => <Item key={item.id}>{item.id === 'int8' ? 'INT8' : 'Original'}</Item>}
    </Picker>
);
