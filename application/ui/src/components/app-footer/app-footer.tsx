import { ActionButton, DialogTrigger, Divider, Flex, Icon, View } from '@geti-ui/ui';
import { Manifest } from '@geti-ui/ui/icons';

import { JobStatus } from '../../features/jobs/footer/job-status';
import { RestartRequiredBanner } from '../../features/plugins/restart-required-banner';
import { useRestartState } from '../../features/plugins/restart-state';
import { LogsDialog } from '../../features/logs/logs-dialog';

export const AppFooter = ({ gridArea = 'footer' }: { gridArea?: string }) => {
    const { restartRequired } = useRestartState();

    return (
        <View
            gridArea={gridArea}
            borderTopColor={'gray-300'}
            borderTopWidth={'thin'}
            borderBottomColor={'gray-75'}
            borderBottomWidth={'thin'}
            paddingX='size-100'
            paddingY='size-25'
        >
            <Flex alignItems={'center'} justifyContent='space-between' gap='size-100' height='100%'>
                <Flex alignItems={'center'} height='100%' gap='size-100' flex={1} minWidth={0}>
                    <View overflow={'hidden'}>
                        <DialogTrigger type='fullscreen'>
                            <ActionButton
                                isQuiet
                                UNSAFE_style={{
                                    paddingRight: 'var(--spectrum-global-dimension-size-100)',
                                }}
                            >
                                <Icon>
                                    <Manifest />
                                </Icon>
                                Logs
                            </ActionButton>
                            {(close) => <LogsDialog close={close} />}
                        </DialogTrigger>
                    </View>
                    <Divider orientation='vertical' size='S' />
                    <JobStatus />
                </Flex>
                {restartRequired ? (
                    <View maxWidth='72ch' minWidth={0}>
                        <RestartRequiredBanner />
                    </View>
                ) : null}
            </Flex>
        </View>
    );
};
