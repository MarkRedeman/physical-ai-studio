import { Button, Flex, View } from '@geti-ui/ui';

import { useProjectId } from '../../features/projects/use-project';
import { Button, ButtonGroup, Flex, Item, TabList, Tabs, View } from '@geti/ui';
import { useParams } from 'react-router';

import { Preview } from '../../features/robots/environment-form/preview';
import { EnvironmentFormProvider, EnvironmentFormState } from '../../features/robots/environment-form/provider';
import { useEnvironment } from '../../features/robots/use-environment';
import { paths } from '../../router';

const ConnectButton = () => {
    return (
        <div
            style={{
                display: 'flex',
                flex: '0 0 auto',
                borderBottom: 'var(--spectrum-alias-border-size-thick) solid var(--spectrum-global-color-gray-300)',
            }}
        >
            <ButtonGroup>
                <Button variant='secondary'>Connect</Button>
            </ButtonGroup>
        </div>
    );
};

const Header = () => {
    const { project_id } = useProjectId();

    return (
        <Flex width='100%'>
            <View
                width='100%'
                borderBottomColor={'gray-400'}
                borderBottomWidth={'thin'}
                padding='size-200'
                margin={'size-200'}
                marginBottom={'size-200'}
                marginTop={'size-100'}
            >
                <Flex justifyContent={'end'} width='100%'>
                    <Button href={paths.project.datasets.index({ project_id })} variant='secondary'>
                        Record dataset
                    </Button>
                    <ConnectButton />
                </Flex>
            </View>
        </Flex>
    );
};

export const EnvironmentShow = () => {
    const params = useParams<{ project_id: string; environment_id: string }>() as {
        project_id: string;
        environment_id: string;
    };

    const environment = useEnvironment();

    const environmentForm: EnvironmentFormState = {
        id: environment.id,
        name: environment.name,
        camera_ids: environment.cameras?.map(({ id }) => id) ?? [],
        robots:
            environment.robots?.map((robot) => {
                return {
                    robot_id: robot.robot.id,
                    teleoperator:
                        robot.tele_operator.type === 'robot'
                            ? {
                                  type: 'robot',
                                  robot_id: robot.tele_operator.robot_id,
                              }
                            : { type: 'none' },
                };
            }) ?? [],
    };
    return (
        <EnvironmentFormProvider environment={environmentForm}>
            <Header />
            <Preview />
        </EnvironmentFormProvider>
    );
};
