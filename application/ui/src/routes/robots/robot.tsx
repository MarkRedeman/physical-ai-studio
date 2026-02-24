import { Button, ButtonGroup, Flex, Item, TabList, TabPanels, Tabs } from '@geti/ui';
import { Outlet, useLocation, useParams } from 'react-router-dom';

import { $api } from '../../api/client';
import { IdentifyRobotButton } from '../../features/robots/identify-robot-button';
import { useRobot } from '../../features/robots/use-robot';
import { paths } from '../../router';
import { getPathSegment } from '../../utils';

const IdentifyRobot = () => {
    return null;
    // const robot = useRobot();
    // const { data: robots } = $api.useSuspenseQuery('get', '/api/hardware/robots');
    // const hardwareRobot = robots.find(({ serial_id }) => serial_id === robot.serial_id);

    // return <IdentifyRobotButton port_id={hardwareRobot?.port ?? ''} />;
};

export const Robot = () => {
    const { pathname } = useLocation();
    const params = useParams<{ project_id: string; robot_id: string }>() as {
        project_id: string;
        robot_id: string;
    };

    const selectedKey = getPathSegment(pathname, 5);
    return (
        <Tabs aria-label='Robot configuration navigation' selectedKey={selectedKey} height='100%'>
            <Flex>
                <TabList
                    width='100%'
                    UNSAFE_style={{
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item key={paths.project.robots.controller(params)} href={paths.project.robots.controller(params)}>
                        Robot controller
                    </Item>
                    <Item
                        key={paths.project.robots.calibration(params)}
                        href={paths.project.robots.calibration(params)}
                    >
                        Calibration
                    </Item>
                    <Item
                        key={paths.project.robots.setupMotors(params)}
                        href={paths.project.robots.setupMotors(params)}
                    >
                        Setup motors
                    </Item>
                </TabList>
                <div
                    style={{
                        display: 'flex',
                        flex: '0 0 auto',
                        borderBottom:
                            'var(--spectrum-alias-border-size-thick) solid var(--spectrum-global-color-gray-300)',
                    }}
                >
                    <ButtonGroup isHidden={selectedKey !== paths.project.robots.controller(params)}>
                        <IdentifyRobot />
                        <Button variant='secondary'>Connect</Button>
                    </ButtonGroup>
                </div>
            </Flex>
            <TabPanels>
                <Item key={paths.project.robots.controller(params)}>
                    <Outlet />
                </Item>
                <Item key={paths.project.robots.calibration(params)}>
                    <Outlet />
                </Item>
                <Item key={paths.project.robots.setupMotors(params)}>
                    <Outlet />
                </Item>
            </TabPanels>
        </Tabs>
    );
};
