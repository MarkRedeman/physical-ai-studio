// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, Button, DialogTrigger, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import type { components } from '../../../api/openapi-spec';

export const DeleteVariant = ({
    modelId,
    projectId,
    backend,
    variant,
}: {
    modelId: string;
    projectId: string;
    backend: components['schemas']['physicalai__export__backends__ExportBackend'];
    variant: string;
}) => {
    const remove = $api.useMutation('delete', '/api/models/{model_id}/exports/{backend}/variants/{variant}', {
        meta: {
            invalidates: [
                ['get', '/api/models/{model_id}', { params: { path: { model_id: modelId } } }],
                ['get', '/api/projects/{project_id}/models', { params: { path: { project_id: projectId } } }],
            ],
        },
    });
    return (
        <>
            <DialogTrigger>
                <Button variant='negative' isDisabled={remove.isPending}>
                    Delete variant
                </Button>
                {(close) => (
                    <AlertDialog
                        title={`Delete ${variant}?`}
                        variant='destructive'
                        primaryActionLabel='Delete'
                        cancelLabel='Cancel'
                        onPrimaryAction={() => {
                            remove.mutate({ params: { path: { model_id: modelId, backend, variant } } });
                            close();
                        }}
                    >
                        This removes only this export variant. The model record, training files, and other variants are
                        kept.
                    </AlertDialog>
                )}
            </DialogTrigger>
            {remove.isError && (
                <div role='alert'>
                    <Text>{getApiErrorMessage(remove.error) ?? 'Could not delete the variant.'}</Text>
                </div>
            )}
        </>
    );
};
