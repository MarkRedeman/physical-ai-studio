import { Content, Flex, Heading, IllustratedMessage, Text, View } from '@geti-ui/ui';

import { RobotViewer } from '../controller/robot-viewer';
import { useRobotCatalogDefinitionQuery } from '../robot-catalog.hooks';
import { ReactComponent as RobotIllustration } from './../../../assets/illustrations/INTEL_08_NO-TESTS.svg';
import { LiveConnectPanel } from './live-connect/panel';
import { useRobotForm } from './provider';

const EmptyPreview = () => {
    return (
        <IllustratedMessage>
            <RobotIllustration />

            <Flex direction='column' gap='size-200'>
                <Content>
                    <Text>
                        Choose the robot you&apos; like to add using the form on the left. After connecting the robot,
                        the preview will appear here.
                    </Text>
                </Content>
                <Heading>Setup your new robot</Heading>
            </Flex>
        </IllustratedMessage>
    );
};

export const Preview = () => {
    const { activeType } = useRobotForm();

    return (
        <View backgroundColor='gray-200' height='100%'>
            <Flex direction='column' height='100%' minHeight={0}>
                <View flex={1} minHeight={0}>
                    {activeType != null ? <ModelPreview robotType={activeType} /> : <EmptyPreview />}
                </View>
                <LiveConnectPanel />
            </Flex>
        </View>
    );
};

const ModelPreview = ({ robotType }: { robotType: string }) => {
    const { data: definition } = useRobotCatalogDefinitionQuery(robotType);
    return definition.urdf_path === null ? (
        <Text>No 3D model supplied by this plugin.</Text>
    ) : (
        <RobotViewer robot={{ type: robotType }} />
    );
};
