// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Content, ContextualHelp, Flex, Heading, Text } from '@geti-ui/ui';
import { createSearchParams, useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import type { components, SchemaModel } from '../../../api/openapi-spec';
import { paths } from '../../../paths';
import { getDefaultInferenceDevice } from '../backend-selection/backend-selection';
import { ExportVariant } from './variant-selection';

export const RunVariant = ({
    model,
    backend,
    variant,
}: {
    model: SchemaModel;
    backend: components['schemas']['physicalai__export__backends__ExportBackend'];
    variant: ExportVariant;
}) => {
    const navigate = useNavigate();
    const {
        data: devices = [],
        isPending,
        isError,
        refetch,
    } = $api.useQuery('get', '/api/system/devices/inference', {}, { enabled: variant.can_run_in_studio });
    const device = getDefaultInferenceDevice(devices, backend);
    const unavailableReason = !variant.can_run_in_studio
        ? 'This export cannot run in Studio.'
        : isError
          ? 'Could not load inference devices. Retry to check available hardware.'
          : !isPending && device === undefined
            ? 'No compatible inference device is available for this backend.'
            : null;

    const run = () => {
        if (!variant.can_run_in_studio || isPending || isError || device === undefined || model.id === undefined)
            return;
        navigate({
            pathname: paths.project.models.inference({ project_id: model.project_id, model_id: model.id, backend }),
            search: createSearchParams({ device: device.device, variant: variant.id }).toString(),
        });
    };

    return (
        <Flex alignItems='center' gap='size-100'>
            <Button
                variant='accent'
                onPress={run}
                isDisabled={
                    !variant.can_run_in_studio || isPending || isError || device === undefined || model.id === undefined
                }
            >
                Run
            </Button>
            {unavailableReason !== null && (
                <ContextualHelp aria-label='Why Run is unavailable' variant='info'>
                    <Heading>Studio inference</Heading>
                    <Content>
                        <Text>{unavailableReason}</Text>
                    </Content>
                </ContextualHelp>
            )}
            {variant.can_run_in_studio && isError && (
                <Button variant='secondary' onPress={() => void refetch()}>
                    Retry devices
                </Button>
            )}
        </Flex>
    );
};
