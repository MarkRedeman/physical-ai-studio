import { Button, Flex, Text, View } from '@geti-ui/ui';

import { useRestartServerMutation } from './plugins.hooks';

export const RestartRequiredBanner = () => {
    const restartMutation = useRestartServerMutation();
    const isRestarting = restartMutation.isPending;

    const restart = async () => {
        try {
            await restartMutation.mutateAsync({});
        } catch {
            // The server may die before responding; treat errors as "restarting".
        }
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
                <Text>
                    {isRestarting
                        ? 'Server is restarting. Please wait a moment and reload the page once it is back.'
                        : 'Restart the server to activate the plugin changes.'}
                </Text>
                <Button variant='primary' isDisabled={isRestarting} onPress={restart}>
                    {isRestarting ? 'Restarting…' : 'Restart server'}
                </Button>
            </Flex>
        </View>
    );
};
