import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaExportVariantDetail, SchemaModel, SchemaModelVariantAction } from '../../../api/openapi-spec';
import { MutationMeta } from '../../../query-client/query-client.interface';
import { INFERENCE_BACKENDS } from '../inference-backends';

export const DeleteExportDialog = ({
    model,
    backend,
    variants,
    variant,
    close,
}: {
    model: SchemaModel;
    backend: SchemaModelVariantAction['backend'];
    variants: SchemaExportVariantDetail[];
    variant?: SchemaExportVariantDetail;
    close: () => void;
}) => {
    const options = {
        meta: {
            invalidates: [
                ['get', '/api/models/{model_id}', { params: { path: { model_id: model.id! } } }],
                ['get', '/api/projects/{project_id}/models', { params: { path: { project_id: model.project_id } } }],
            ],
        } satisfies MutationMeta,
        onSuccess: close,
    };
    const removeBackend = $api.useMutation('delete', '/api/models/{model_id}/exports/{backend}', options);
    const removeVariant = $api.useMutation(
        'delete',
        '/api/models/{model_id}/exports/{backend}/variants/{variant}',
        options
    );
    const pending = removeBackend.isPending || removeVariant.isPending;
    const error = removeBackend.error ?? removeVariant.error;
    const selected = variant ? [variant] : variants;
    const megabytes = selected.reduce((total, item) => total + item.size_bytes, 0) / 1024 ** 2;
    const size = megabytes >= 1024 ? `${(megabytes / 1024).toFixed(1)} GB` : `${megabytes.toFixed(1)} MB`;
    const label = INFERENCE_BACKENDS[backend].label;
    const title = variant ? 'Remove export' : 'Remove backend';
    return (
        <Dialog>
            <Heading>{title}</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        {variant
                            ? `Permanently remove ${label} · ${variant.id === 'int8' ? 'INT8' : 'Original'}? ` +
                              'Other exports stay available.'
                            : `Permanently remove ${label} and all its variants?`}
                    </Text>
                    {!variant && (
                        <Text>
                            Exports removed:{' '}
                            {selected.map((item) => (item.id === 'int8' ? 'INT8' : 'Original')).join(', ')}.
                        </Text>
                    )}
                    <Text>
                        Approximately {size} will be freed. The training checkpoint, training history, and other
                        backends are kept.
                    </Text>
                    <Text>
                        Unload this export from inference and wait for export jobs to finish before removing it.
                    </Text>
                    {error && (
                        <div role='alert'>
                            {getApiErrorMessage(error) ?? 'Could not remove the export. Please try again.'}
                        </div>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' isDisabled={pending} onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='negative'
                    isPending={pending}
                    onPress={() => {
                        const path = { model_id: model.id!, backend };
                        if (variant) removeVariant.mutate({ params: { path: { ...path, variant: variant.id } } });
                        else removeBackend.mutate({ params: { path } });
                    }}
                >
                    {title}
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
