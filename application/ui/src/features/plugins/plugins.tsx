import { useState } from 'react';

import { Badge, Button, Card, Flex, Heading, Link, Text, View } from '@geti-ui/ui';
import { clsx } from 'clsx';

import { getApiErrorMessage, isResourceInUseError } from '../../api/errors';
import { SchemaPluginResponse, SchemaPluginRobotResponse } from '../../api/openapi-spec';
import { useInstallPluginMutation, usePluginsQuery, useUninstallPluginMutation } from './plugins.hooks';
import { RestartRequiredBanner } from './restart-required-banner';

import classes from './plugins.module.css';

const ROLE_CLASS_NAMES = {
    follower: classes.roleFollower,
    leader: classes.roleLeader,
} as const;

const RoleBadge = ({ role }: { role: SchemaPluginRobotResponse['role'] }) => (
    <Badge variant='neutral' UNSAFE_className={clsx(classes.roleBadge, ROLE_CLASS_NAMES[role])}>
        {role}
    </Badge>
);

const PluginRobots = ({ robots }: { robots: SchemaPluginRobotResponse[] }) => {
    if (robots.length === 0) {
        return <Text>Robots are discovered after installation.</Text>;
    }
    return (
        <Flex gap='size-50' wrap>
            {robots.map((robot) => (
                <View key={robot.type} padding='size-50' UNSAFE_className={classes.robotChip}>
                    <Flex alignItems='center' gap='size-50'>
                        <RoleBadge role={robot.role} />
                        <Text UNSAFE_className={classes.robotName}>{robot.display_name}</Text>
                    </Flex>
                </View>
            ))}
        </Flex>
    );
};

const PluginCard = ({
    plugin,
    onInstall,
    onUninstall,
}: {
    plugin: SchemaPluginResponse;
    onInstall: (pluginId: string) => void;
    onUninstall: (pluginId: string) => void;
}) => {
    const isInstalled = plugin.installed;
    const isInUse = plugin.in_use_robot_count > 0;

    return (
        <Card UNSAFE_className={classes.card} aria-label={plugin.name}>
            <Flex direction='column' gap='size-150' height='100%'>
                <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
                    <Heading level={3}>{plugin.name}</Heading>
                    {isInstalled ? (
                        <Badge variant='positive'>v{plugin.installed_version}</Badge>
                    ) : (
                        <Badge variant='neutral'>Available</Badge>
                    )}
                </Flex>
                <Text>{plugin.description}</Text>
                <View flex={1}>
                    <PluginRobots robots={plugin.robots} />
                </View>
                {isInstalled && isInUse ? (
                    <Text UNSAFE_className={classes.inUse}>
                        In use by {plugin.in_use_robot_count} robot{plugin.in_use_robot_count === 1 ? '' : 's'}
                    </Text>
                ) : null}
                <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
                    {plugin.repo_url ? (
                        <Link href={plugin.repo_url} target='_blank' rel='noreferrer'>
                            GitHub
                        </Link>
                    ) : (
                        <View />
                    )}
                    {isInstalled ? (
                        <Button variant='secondary' isDisabled={isInUse} onPress={() => onUninstall(plugin.id)}>
                            Uninstall
                        </Button>
                    ) : (
                        <Button variant='primary' onPress={() => onInstall(plugin.id)}>
                            Install
                        </Button>
                    )}
                </Flex>
            </Flex>
        </Card>
    );
};

export const PluginsView = () => {
    const pluginsQuery = usePluginsQuery();
    const installMutation = useInstallPluginMutation();
    const uninstallMutation = useUninstallPluginMutation();
    const [restartRequired, setRestartRequired] = useState(false);

    const plugins = pluginsQuery.data;
    const installed = plugins.filter((plugin) => plugin.installed);
    const available = plugins.filter((plugin) => !plugin.installed);

    const install = async (pluginId: string) => {
        try {
            await installMutation.mutateAsync({ params: { path: { plugin_id: pluginId } } });
            setRestartRequired(true);
        } catch (error) {
            if (isResourceInUseError(error)) {
                return;
            }
            console.error(`Failed to install plugin: ${pluginId}`, error);
        }
    };

    const uninstall = async (pluginId: string) => {
        try {
            await uninstallMutation.mutateAsync({ params: { path: { plugin_id: pluginId } } });
            setRestartRequired(true);
        } catch (error) {
            const message = getApiErrorMessage(error);
            if (isResourceInUseError(error)) {
                return;
            }
            console.error(`Failed to uninstall plugin: ${pluginId}`, error, message);
        }
    };

    return (
        <Flex direction='column' gap='size-300' minHeight={0} height='100%'>
            <Heading>Plugins</Heading>
            {restartRequired ? <RestartRequiredBanner /> : null}
            {installed.length > 0 ? (
                <Flex direction='column' gap='size-150'>
                    <Heading level={2}>Installed</Heading>
                    <Flex gap='size-200' wrap>
                        {installed.map((plugin) => (
                            <PluginCard key={plugin.id} plugin={plugin} onInstall={install} onUninstall={uninstall} />
                        ))}
                    </Flex>
                </Flex>
            ) : null}
            {available.length > 0 ? (
                <Flex direction='column' gap='size-150'>
                    <Heading level={2}>Available</Heading>
                    <Flex gap='size-200' wrap>
                        {available.map((plugin) => (
                            <PluginCard key={plugin.id} plugin={plugin} onInstall={install} onUninstall={uninstall} />
                        ))}
                    </Flex>
                </Flex>
            ) : null}
        </Flex>
    );
};
