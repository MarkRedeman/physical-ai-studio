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
            <View padding='size-400' height='100%'>
                <Tabs aria-label='Environment' height='100%'>
                    <Flex>
                        <TabList
                            width='100%'
                            UNSAFE_style={{
                                '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                            }}
                        >
                            <Item
                                key={paths.project.environments.overview(params)}
                                href={paths.project.environments.overview(params)}
                            >
                                Overview
                            </Item>
                            <Item
                                key={paths.project.environments.datasets(params)}
                                href={paths.project.environments.datasets(params)}
                            >
                                Datasets
                            </Item>
                            <Item
                                key={paths.project.environments.models(params)}
                                href={paths.project.environments.models(params)}
                            >
                                Model
                            </Item>
                        </TabList>
                        <ConnectButton />
                    </Flex>
                    <Preview />
                </Tabs>
            </View>
        </EnvironmentFormProvider>
    );
};
