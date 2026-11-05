import { Divider, Flex, Heading, Text, View } from '@geti-ui/ui';

import type { components, SchemaModel } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, type InferenceBackendConfig } from '../inference-backends';
import { VariantActions } from './variant-actions';
import { ExportVariant } from './variant-selection';

type ExportBackend = components['schemas']['physicalai__export__backends__ExportBackend'];

export const InferenceBackendLogo = ({
    backend,
    isAvailable,
    compact = false,
}: {
    backend: InferenceBackendConfig;
    isAvailable: boolean;
    compact?: boolean;
}) => {
    const Logo = backend.logo;
    return (
        <Flex alignItems='center' gap='size-200' UNSAFE_style={isAvailable ? undefined : { opacity: 0.4 }}>
            <Logo height={compact ? '32px' : '50px'} width={compact ? '32px' : '50px'} />
            <Flex direction='column' gap='size-10'>
                <Heading level={4} margin={0}>
                    {backend.label}
                </Heading>
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
                            <InferenceBackendLogo backend={backend} isAvailable compact />
                            <VariantActions
                                model={model}
                                backend={backendType}
                                variant={variant}
                                onOptimize={onOptimize}
                            />
                        </Flex>
                        <Heading level={4} margin={0}>
                            {variant.label}
                        </Heading>
                        <Divider size='S' />
                        <Flex gap='size-300' wrap>
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
                        </Flex>
                    </Flex>
                </Flex>
            </View>
        </section>
    );
};
