import { useState } from 'react';

import { Divider, Flex, Item, Picker, Text, View } from '@geti-ui/ui';

import type { components } from '../../../api/openapi-spec';
import { Box } from '../shared/box';
import { FeatureRow } from './feature-row';

type ExportDetail = components['schemas']['BackendExportDetail'];
type IOFeature = components['schemas']['IOFeature'];

const ModelInputInterface = ({ inputFeatures }: { inputFeatures: IOFeature[] }) => {
    return (
        <Flex direction='column' gap='size-100' width='size-3600'>
            <Text UNSAFE_style={{ fontWeight: 600 }}>Inputs ({inputFeatures.length})</Text>
            <Flex direction={'column'} gap='size-200'>
                {inputFeatures.map((feature) => {
                    return <FeatureRow key={feature.name} feature={feature} />;
                })}
            </Flex>
        </Flex>
    );
};

const ModelOutputInterface = ({ outputFeatures }: { outputFeatures: IOFeature[] }) => {
    return (
        <Flex direction='column' gap='size-100' width='size-3600'>
            <Text UNSAFE_style={{ fontWeight: 600 }}>Outputs ({outputFeatures.length})</Text>
            <Flex direction={'column'} gap='size-200'>
                {outputFeatures.map((feature) => {
                    return <FeatureRow key={feature.name} feature={feature} />;
                })}
            </Flex>
        </Flex>
    );
};

export const ModelInterface = ({ exports }: { exports: ExportDetail[] }) => {
    const [selected, setSelected] = useState<string>();
    const exportsWithIoSpec = exports
        .flatMap((backend) =>
            backend.variants?.length
                ? backend.variants.map((variant) => ({
                      id: `${backend.type}/${variant.id}`,
                      type: backend.type,
                      label: `${backend.type} · ${variant.label}`,
                      io_spec: variant.io_spec,
                  }))
                : [
                      {
                          id: `${backend.type}/original`,
                          type: backend.type,
                          label: `${backend.type} · Original`,
                          io_spec: backend.io_spec,
                      },
                  ]
        )
        .filter(({ io_spec }) => io_spec != null);

    const mainFormat =
        exportsWithIoSpec.find((entry) => entry.id === selected) ??
        exportsWithIoSpec.find((e) => {
            return e.type === 'torch';
        }) ??
        exportsWithIoSpec.at(0);

    if (exportsWithIoSpec.length === 0 || mainFormat === undefined) {
        return null;
    }

    const inputFeatures = mainFormat.io_spec?.input_features ?? [];
    const outputFeatures = mainFormat.io_spec?.output_features ?? [];

    return (
        <View gridArea='model-interface'>
            <Box
                title='Model interface'
                content={
                    <Flex direction='column' gap='size-200'>
                        {exportsWithIoSpec.length > 1 && (
                            <Picker
                                label='Export interface'
                                items={exportsWithIoSpec}
                                selectedKey={mainFormat.id}
                                onSelectionChange={(key) => key !== null && setSelected(String(key))}
                            >
                                {(entry) => <Item key={entry.id}>{entry.label}</Item>}
                            </Picker>
                        )}
                        <Flex direction='row' gap='size-400'>
                            <ModelInputInterface inputFeatures={inputFeatures} />

                            <Divider size='S' orientation='vertical' />

                            <ModelOutputInterface outputFeatures={outputFeatures} />
                        </Flex>
                    </Flex>
                }
            />
        </View>
    );
};
