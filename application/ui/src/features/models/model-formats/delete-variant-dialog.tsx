// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Content, Dialog, Divider, Heading, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import type { components } from '../../../api/openapi-spec';

export const DeleteVariantDialog = ({
    modelId,
    projectId,
    backend,
    variant,
    close,
}: {
    modelId: string;
    projectId: string;
    backend: components['schemas']['physicalai__export__backends__ExportBackend'];
    variant: string;
    close: () => void;
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
        <Dialog>
            <Heading>Delete {variant}?</Heading>
            <Divider />
            <Content>
                <Text>
                    This removes only this export variant. The model record, training files, and other variants are
                    kept.
                </Text>
                {remove.isError && (
                    <div role='alert'>
                        <Text>{getApiErrorMessage(remove.error) ?? 'Could not delete the variant.'}</Text>
                    </div>
                )}
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close} isDisabled={remove.isPending}>
                    Cancel
                </Button>
                <Button
                    variant='negative'
                    isDisabled={remove.isPending}
                    onPress={() =>
                        remove.mutate(
                            { params: { path: { model_id: modelId, backend, variant } } },
                            { onSuccess: close }
                        )
                    }
                >
                    Delete
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
