import { Button, Flex, Text, View } from '@geti-ui/ui';

import { useRestartServerMutation, useRestartState } from './restart-state';

const statusText: Record<string, string> = {
    idle: 'Restart the server to activate the plugin changes.',
    requesting: 'Requesting server restart…',
    waiting_for_down: 'Waiting for server to go down…',
    waiting_for_up: 'Waiting for server startup…',
    failed: 'Could not confirm restart from health checks. You can retry.',
};

export const RestartRequiredBanner = () => {
    const restartMutation = useRestartServerMutation();
    const isRestarting = restartMutation.isPending;
    const { restartRequired, openRestartPrompt } = useRestartState();

    const restart = async () => {
        openRestartPrompt();
    };

    if (restartRequired === false) {
        return null;
    }

    return (
        <View backgroundColor={'yellow-400'} paddingX='size-400'>
            <Flex
                alignItems='center'
                justifyContent='space-between'
                gap='size-100'
                UNSAFE_style={{
                    color: 'black',
                }}
            >
                <Text>{statusText[restartMutation.restartStatus]}</Text>
                <Button
                    variant='primary'
                    style='fill'

                    isDisabled={isRestarting}
                    onPress={restart}
                    UNSAFE_style={{
                        minHeight: '24px',
                        paddingInline: 'var(--spectrum-global-dimension-size-200)',
                        borderRadius: 0,
                    }}
                >
                    {isRestarting ? 'Restarting…' : 'Restart server'}
                </Button>
            </Flex>
        </View>
    );
};
