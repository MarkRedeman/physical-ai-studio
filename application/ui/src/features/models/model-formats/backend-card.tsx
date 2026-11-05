import { Button, Divider, Flex, Heading, Text, View } from '@geti-ui/ui';

import type { components, SchemaModel } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, type InferenceBackendConfig } from '../inference-backends';
import { RunVariant } from './run-variant';
import { VariantActions } from './variant-actions';
import { ExportVariant } from './variant-selection';

import classes from './model-formats.module.css';

type ExportBackend = components['schemas']['physicalai__export__backends__ExportBackend'];

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
            <Flex direction='column' gap='size-10'>
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

const formatSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(1)} MB`;
};

interface BackendCardProps {
    backendType: ExportBackend;
    model: SchemaModel;
    variant: ExportVariant;
    onOptimize: () => void;
}

export const BackendCard = ({ backendType, model, variant, onOptimize }: BackendCardProps) => {
    const backend = INFERENCE_BACKENDS[backendType];
    const canOptimize = variant.available_operations.some((operation) => operation.available);

    return (
        <section aria-label={`${backend.label} · ${variant.label}`}>
            <View
                backgroundColor='gray-50'
                padding='size-200'
                borderRadius='medium'
                borderWidth='thin'
                borderColor='gray-100'
                height='100%'
            >
                <Flex direction='column' justifyContent='space-between' gap='size-200' height='100%'>
                    <Flex direction='column' gap='size-150'>
                        <Flex justifyContent='space-between' alignItems='center' gap='size-100'>
                            <InferenceBackendLogo
                                backend={backend}
                                isAvailable
                                compact
                                variantLabel={variant.id === 'original' ? undefined : variant.label}
                            />
                            <VariantActions
                                model={model}
                                backend={backendType}
                                variant={variant}
                                onOptimize={onOptimize}
                            />
                        </Flex>
                        <Divider size='S' />
                        <Flex gap='size-300' alignItems='end' wrap>
                            <Flex direction='column' gap='size-50'>
                                <Text>Size</Text>
                                <Text>{formatSize(variant.size_bytes)}</Text>
                            </Flex>
                            <Flex direction='column' gap='size-50'>
                                <Text>Weights</Text>
                                <Text>
                                    {variant.weight_compression === 'int8_sym' ? 'INT8 symmetric weights' : 'Original'}
                                </Text>
                            </Flex>
                            <View marginStart='auto'>
                                <Flex alignItems='center' gap='size-100'>
                                    {canOptimize && (
                                        <Button variant='secondary' onPress={() => onOptimize()}>
                                            Optimize
                                        </Button>
                                    )}
                                    <RunVariant model={model} backend={backendType} variant={variant} />
                                </Flex>
                            </View>
                        </Flex>
                    </Flex>
                </Flex>
            </View>
        </section>
    );
};
