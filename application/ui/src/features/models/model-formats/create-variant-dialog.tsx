import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Item, Picker } from '@geti-ui/ui';

import { SchemaModelDetailResponse, SchemaModelVariantAction } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS } from '../inference-backends';
import { VariantJobControls } from './variant-job-controls';

export const CreateVariantDialog = ({
    modelDetail,
    initialAction,
    close,
}: {
    modelDetail: SchemaModelDetailResponse;
    initialAction?: SchemaModelVariantAction;
    close: () => void;
}) => {
    const [actions] = useState(modelDetail.available_actions ?? []);
    const [selection, setSelection] = useState(initialAction ?? actions[0]);
    const target = selection?.operation === 'int8_sym' ? 'int8' : 'original';
    const targetExists = modelDetail.exports.some(
        (backend) => backend.type === selection?.backend && backend.variants?.some((variant) => variant.id === target)
    );
    return (
        <Dialog>
            <Heading>Create model variant</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Picker
                        label='Variant'
                        items={actions}
                        selectedKey={selection && `${selection.backend}/${selection.operation}`}
                        onSelectionChange={(key) => {
                            const action = actions.find((item) => `${item.backend}/${item.operation}` === key);
                            if (action) setSelection(action);
                        }}
                        width='100%'
                    >
                        {(action) => (
                            <Item key={`${action.backend}/${action.operation}`}>
                                {INFERENCE_BACKENDS[action.backend].label} ·{' '}
                                {action.operation === 'int8_sym' ? 'INT8' : 'Original'}
                            </Item>
                        )}
                    </Picker>
                    {selection && (
                        <VariantJobControls
                            key={`${selection.backend}/${selection.operation}`}
                            modelId={modelDetail.model.id!}
                            action={selection}
                            targetExists={targetExists}
                        />
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
