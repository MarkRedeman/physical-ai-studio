import { Heading, IllustratedMessage, Item, TabList, TabPanels, Tabs, View } from '@geti-ui/ui';

import { SchemaModel } from '../../api/openapi-spec';
import { ModelExports } from '../../features/models/exports/model-exports';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { MetricsContent } from './metrics';

import classes from './model-row-content.module.scss';

const ComingSoon = () => {
    return (
        <IllustratedMessage marginY='size-400'>
            <EmptyIllustration height='250px' />
            <Heading>Coming soon</Heading>
        </IllustratedMessage>
    );
};

interface ModelRowContentProps {
    model: SchemaModel;
}

export const ModelRowContent = ({ model }: ModelRowContentProps) => {
    return (
        <View UNSAFE_className={classes.modelRowContent}>
            <Tabs>
                <TabList>
                    <Item key='export'>Export model</Item>
                    <Item key='metrics'>Model Metrics</Item>
                    <Item key='datasets'>Training Datasets</Item>
                </TabList>
                <TabPanels>
                    <Item key='export'>
                        <ModelExports model={model} />
                    </Item>
                    <Item key='metrics'>
                        <MetricsContent modelId={model.id!} />
                    </Item>
                    <Item key='datasets'>
                        <ComingSoon />
                    </Item>
                </TabPanels>
            </Tabs>
        </View>
    );
};
