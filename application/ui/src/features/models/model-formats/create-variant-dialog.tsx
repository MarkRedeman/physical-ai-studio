// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Item, Picker, Text } from '@geti-ui/ui';

import type { SchemaModelDetailResponse } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, isExportBackend } from '../inference-backends';
import { OptimizeModel } from './optimize-model';

export const CreateVariantDialog = ({
    modelDetail,
    close,
    initialSourceId,
}: {
    modelDetail: SchemaModelDetailResponse;
    close: () => void;
    initialSourceId?: string;
}) => {
    const sources = modelDetail.exports.flatMap((backend) => {
        const backendType = backend.type;
        return isExportBackend(backendType)
            ? backend.variants.map((variant) => ({
                  id: `${backendType}/${variant.id}`,
                  backend: backendType,
                  variant,
                  label: `${INFERENCE_BACKENDS[backendType].label} · ${variant.label}`,
              }))
            : [];
    });
    const [sourceId, setSourceId] = useState<string | undefined>(initialSourceId);
    const source = sources.find((item) => item.id === sourceId) ?? sources[0];

    return (
        <Dialog>
            <Heading>Create model variant</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-250'>
                    <Picker
                        label='Source export'
                        items={sources}
                        selectedKey={source?.id}
                        onSelectionChange={(key) => key !== null && setSourceId(String(key))}
                        width='100%'
                    >
                        {(item) => <Item key={item.id}>{item.label}</Item>}
                    </Picker>
                    {source ? (
                        <OptimizeModel
                            key={source.id}
                            modelId={modelDetail.model.id!}
                            backend={source.backend}
                            variant={source.variant.id}
                            operations={source.variant.available_operations}
                        />
                    ) : (
                        <Text>A complete export is required before creating a variant.</Text>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
