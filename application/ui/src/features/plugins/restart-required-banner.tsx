import { Button, Flex, Text, View } from '@geti-ui/ui';

import { useRestartServerMutation } from './plugins.hooks';

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

    const restart = async () => {
        await restartMutation.restartServer();
    };

    return (
        <View
            padding='size-200'
            borderColor='yellow-400'
            borderWidth='thin'
            borderRadius='regular'
            backgroundColor='yellow-100'
        >
            <Flex alignItems='center' justifyContent='space-between' gap='size-200'>
                <Text>{statusText[restartMutation.restartStatus]}</Text>
                <Button variant='primary' isDisabled={isRestarting} onPress={restart}>
                    {isRestarting ? 'Restarting…' : 'Restart server'}
                </Button>
            </Flex>
        </View>
    );
};
