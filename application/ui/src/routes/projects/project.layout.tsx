import { Suspense } from 'react';

import { Flex, Grid, Item, Link, Loading, TabList, Tabs, View } from '@geti/ui';
import { Outlet, useLocation } from 'react-router';

import { ShowLogs } from '../../features/logs/show-logs.component';
import { ProjectsListPanel } from '../../features/projects/menu/projects-list-panel.component';
import { useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';
import { ReactComponent as DatasetIcon } from './../../assets/icons/dataset-icon.svg';
import { ReactComponent as ModelsIcon } from './../../assets/icons/models-icon.svg';
import { ReactComponent as RobotIcon } from './../../assets/icons/robot-icon.svg';

const Header = ({ project_id }: { project_id: string }) => {
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <Link href='/' isQuiet variant='overBackground'>
                    <View marginEnd='size-200' maxWidth={'10ch'}>
                        Physical AI Studio
                    </View>
                </Link>

                <TabList
                    height={'100%'}
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '4px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item
                        textValue='Robot configuration'
                        key={'robots'}
                        href={paths.project.robots.index({ project_id })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <RobotIcon />
                            Robots
                        </Flex>
                    </Item>
                    <Item textValue='Datasets' key={'datasets'} href={paths.project.datasets.index({ project_id })}>
                        <Flex alignItems='center' gap='size-100'>
                            <DatasetIcon />
                            Datasets
                        </Flex>
                    </Item>
                    <Item textValue='Models' key={'models'} href={paths.project.models.index({ project_id })}>
                        <Flex alignItems='center' gap='size-100'>
                            <ModelsIcon />
                            Models
                        </Flex>
                    </Item>
                </TabList>
                <Flex alignItems={'center'} height={'100%'} marginStart='auto' gap='size-100'>
                    <ProjectsListPanel />
                </Flex>
            </Flex>
        </View>
    );
};

const getMainPageInProjectUrl = (pathname: string) => {
    const regexp = /\/projects\/[\w-]*\/([\w-]*)/g;
    const found = [...pathname.matchAll(regexp)];
    if (found.length) {
        const [, main] = found[0];
        if (main === 'cameras' || main === 'environments') {
            return 'robots';
        }
        return main;
    } else {
        return 'datasets';
    }
};

const Footer = () => {
    return (
        <View
            backgroundColor={'gray-50'}
            gridArea={'footer'}
            borderTopColor={'gray-100'}
            borderTopWidth='thin'
            padding='size-50'
        >
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <ShowLogs />
            </Flex>
        </View>
    );
};

export const ProjectLayout = () => {
    const { project_id } = useProjectId();
    const { pathname } = useLocation();

    const pageName = getMainPageInProjectUrl(pathname);

    return (
        <Tabs aria-label='Header navigation' selectedKey={pageName} UNSAFE_style={{ height: '100%', minHeight: 0 }}>
            <Grid
                areas={['header', 'subheader', 'content', 'footer']}
                UNSAFE_style={{
                    gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) min-content 1fr min-content',
                }}
                minHeight={0}
                height={'100%'}
            >
                <Header project_id={project_id} />
                <View gridArea={'content'} maxHeight={'100vh'} minHeight={0} height='100%' overflow={'auto'}>
                    <Suspense fallback={<Loading mode='overlay' />}>
                        <Outlet />
                    </Suspense>
                </View>
                <Footer />
            </Grid>
        </Tabs>
    );
};
