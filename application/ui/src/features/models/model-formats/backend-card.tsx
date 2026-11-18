import { useState } from 'react';

import {
    ActionButton,
    Button,
    DialogContainer,
    Divider,
    Flex,
    Heading,
    Item,
    Menu,
    MenuTrigger,
    Text,
    View,
} from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { fetchClient } from '../../../api/client';
import { SchemaExportVariantDetail, SchemaModel, SchemaModelVariantAction } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, InferenceBackendConfig } from '../inference-backends';
import { RemoveModelDialog } from './remove-model-dialog';
import { RunVariant } from './run-variant';
import { RuntimeExportDialog } from './runtime-export-dialog';

import classes from './model-formats.module.css';

export const InferenceBackendLogo = ({
    backend,
    isAvailable,
    compact = false,
    variantLabel,
}: {
    backend: InferenceBackendConfig;
    isAvailable: boolean;
    compact?: boolean;
    variantLabel?: string;
}) => {
    const Logo = backend.logo;
    return (
        <Flex alignItems='center' gap='size-200' UNSAFE_style={isAvailable ? undefined : { opacity: 0.4 }}>
            <Logo height={compact ? '32px' : '50px'} width={compact ? '32px' : '50px'} />
            <Flex direction='column' gap='size-50'>
                <Flex alignItems='baseline' gap='size-100' wrap>
                    <Heading level={4} margin={0}>
                        {backend.label}
                    </Heading>
                    {variantLabel && <Text UNSAFE_className={classes.variantDescription}>{variantLabel}</Text>}
                </Flex>
                {!compact && <Text UNSAFE_style={{ fontSize: '11px' }}>{backend.description}</Text>}
            </Flex>
        </Flex>
    );
};

export const BackendCard = ({
    backendType,
    model,
    variant,
    action,
    onCreate,
}: {
    backendType: SchemaModelVariantAction['backend'];
    model: SchemaModel;
    variant: SchemaExportVariantDetail;
    action?: SchemaModelVariantAction;
    onCreate: (action: SchemaModelVariantAction) => void;
}) => {
    const [dialog, setDialog] = useState<'runtime' | 'remove' | null>(null);
    const backend = INFERENCE_BACKENDS[backendType];
    const label = variant.id === 'int8' ? 'INT8' : 'Original';
    const megabytes = variant.size_bytes / 1024 ** 2;
    const size = megabytes >= 1024 ? `${(megabytes / 1024).toFixed(1)} GB` : `${megabytes.toFixed(1)} MB`;
    const downloadUrl = fetchClient.PATH('/api/models/{model_id}/exports/{backend}/download', {
        params: { path: { model_id: model.id!, backend: backendType }, query: { variant: variant.id } },
    });
    return (
        <section aria-label={`${backend.label} · ${label}`}>
            <View
                backgroundColor='gray-50'
                padding='size-200'
                borderRadius='medium'
                borderWidth='thin'
                borderColor='gray-100'
                height='100%'
            >
                <Flex direction='column' justifyContent='space-between' gap='size-200' height='100%'>
                    <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
                        <InferenceBackendLogo
                            backend={backend}
                            isAvailable
                            compact
                            variantLabel={variant.id === 'int8' ? label : undefined}
                        />
                        <MenuTrigger>
                            <ActionButton isQuiet aria-label={`Actions for ${backend.label} ${label}`}>
                                <MoreMenu />
                            </ActionButton>
                            <Menu
                                onAction={(key) => {
                                    if (key === 'runtime' || key === 'remove') setDialog(key);
                                }}
                            >
                                <Item key='download' href={downloadUrl} target='_blank' rel='noopener noreferrer'>
                                    Download
                                </Item>
                                {backendType === 'openvino' ? <Item key='runtime'>Runtime export</Item> : null}
                                <Item key='remove'>Remove model</Item>
                            </Menu>
                        </MenuTrigger>
                    </Flex>
                    <Divider size='S' />
                    <Flex alignItems='end' gap='size-200' wrap>
                        <Flex direction='column'>
                            <Text>Size</Text>
                            <Text>{size}</Text>
                        </Flex>
                        <Flex direction='column'>
                            <Text>Weights</Text>
                            <Text>{variant.weight_compression === 'int8_sym' ? 'INT8 symmetric' : 'Original'}</Text>
                        </Flex>
                        <View marginStart='auto'>
                            <Flex gap='size-100' alignItems='center'>
                                {action && (
                                    <Button variant='secondary' onPress={() => onCreate(action)}>
                                        Optimize
                                    </Button>
                                )}
                                <RunVariant model={model} backend={backendType} variant={variant.id} />
                            </Flex>
                        </View>
                    </Flex>
                </Flex>
            </View>
            <DialogContainer onDismiss={() => setDialog(null)}>
                {dialog === 'runtime' && (
                    <RuntimeExportDialog
                        model={model}
                        backend={backendType}
                        variant={variant.id}
                        close={() => setDialog(null)}
                    />
                )}
                {dialog === 'remove' && (
                    <RemoveModelDialog
                        model={model}
                        backend={backendType}
                        variant={variant}
                        close={() => setDialog(null)}
                    />
                )}
            </DialogContainer>
        </section>
    );
};
