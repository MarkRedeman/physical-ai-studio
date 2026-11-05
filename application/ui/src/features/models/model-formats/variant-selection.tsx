// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Item, Picker } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import type { components } from '../../../api/openapi-spec';

export type ExportVariant = components['schemas']['ExportVariantDetail'];

export const useVariantSelection = (
    modelId: string | undefined,
    backend: string,
    initialVariant?: string,
    detail?: components['schemas']['ModelDetailResponse']
) => {
    const query = $api.useQuery(
        'get',
        '/api/models/{model_id}',
        { params: { path: { model_id: modelId ?? '' } } },
        { enabled: modelId !== undefined && detail === undefined }
    );
    const data = detail ?? query.data;
    const [selection, setSelection] = useState<{ modelId: string | undefined; backend: string; variant: string }>();
    const variants = data?.exports.find((item) => item.type === backend)?.variants ?? [];
    const requested =
        selection !== undefined && selection.modelId === modelId && selection.backend === backend
            ? selection.variant
            : initialVariant;
    const selected =
        variants.find((variant) => variant.id === requested) ??
        (data?.model.snapflow_enabled ? variants.find((variant) => variant.id === 'snapflow') : undefined) ??
        variants.find((variant) => variant.id === 'original') ??
        variants[0];
    return {
        variants,
        selected,
        variant: selected?.id ?? 'original',
        isPending: detail === undefined && query.isPending,
        isError: detail === undefined && query.isError,
        setVariant: (variant: string) => setSelection({ modelId, backend, variant }),
    };
};

export const VariantSelection = ({
    variants,
    variant,
    onChange,
    isDisabled = false,
}: {
    variants: ExportVariant[];
    variant: string;
    onChange: (variant: string) => void;
    isDisabled?: boolean;
}) => (
    <Picker
        label='Variant'
        items={variants}
        selectedKey={variant}
        onSelectionChange={(key) => key !== null && onChange(String(key))}
        isDisabled={isDisabled}
    >
        {(item) => <Item key={item.id}>{item.label}</Item>}
    </Picker>
);
