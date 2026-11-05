// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { ActionButton, DialogContainer, Item, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { fetchClient } from '../../../api/client';
import type { components, SchemaModel } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS } from '../inference-backends';
import { DeleteVariantDialog } from './delete-variant-dialog';
import { RuntimeExportDialog } from './runtime-export-dialog';
import { ExportVariant } from './variant-selection';

export const VariantActions = ({
    model,
    backend,
    variant,
    onOptimize,
}: {
    model: SchemaModel;
    backend: components['schemas']['physicalai__export__backends__ExportBackend'];
    variant: ExportVariant;
    onOptimize: () => void;
}) => {
    const [dialog, setDialog] = useState<'runtime' | 'delete' | null>(null);
    const canOptimize = variant.available_operations.some((operation) => operation.available);
    const downloadUrl = fetchClient.PATH('/api/models/{model_id}/exports/{backend}/download', {
        params: { path: { model_id: model.id!, backend }, query: { variant: variant.id } },
    });

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={`Actions for ${INFERENCE_BACKENDS[backend].label} ${variant.label}`}>
                    <MoreMenu />
                </ActionButton>
                <Menu
                    onAction={(key) => {
                        if (key === 'optimize') onOptimize();
                        if (key === 'runtime' || key === 'delete') setDialog(key);
                    }}
                >
                    {canOptimize ? <Item key='optimize'>Optimize</Item> : null}
                    <Item key='download' href={downloadUrl} target='_blank' rel='noopener noreferrer'>
                        Download
                    </Item>
                    {backend === 'openvino' ? <Item key='runtime'>Runtime export</Item> : null}
                    {variant.id !== 'original' ? <Item key='delete'>Delete</Item> : null}
                </Menu>
            </MenuTrigger>
            <Suspense fallback={null}>
                <DialogContainer onDismiss={() => setDialog(null)}>
                    {dialog === 'runtime' ? (
                        <RuntimeExportDialog
                            model={model}
                            backend={backend}
                            initialVariant={variant.id}
                            fixedVariant
                            close={() => setDialog(null)}
                        />
                    ) : dialog === 'delete' ? (
                        <DeleteVariantDialog
                            modelId={model.id!}
                            projectId={model.project_id}
                            backend={backend}
                            variant={variant.id}
                            close={() => setDialog(null)}
                        />
                    ) : null}
                </DialogContainer>
            </Suspense>
        </>
    );
};
