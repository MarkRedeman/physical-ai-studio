import { useState } from 'react';

import { Badge, Button, Card, Flex, Heading, Link, Text, toast, View } from '@geti-ui/ui';
import { clsx } from 'clsx';

import { getApiErrorMessage, isResourceInUseError } from '../../api/errors';
import { SchemaPluginExtensionResponse, SchemaPluginResponse, SchemaPluginRobotResponse } from '../../api/openapi-spec';
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

const ExtensionRow = ({
    extension,
    isBusy,
    busyId,
    onInstall,
    onUninstall,
}: {
    extension: SchemaPluginExtensionResponse;
    isBusy: boolean;
    busyId: string | undefined;
    onInstall: (pluginId: string) => void;
    onUninstall: (pluginId: string) => void;
}) => {
    const isThisBusy = busyId === extension.id && isBusy;
    return (
        <View padding='size-100' UNSAFE_className={classes.extensionRow}>
            <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
                <Flex direction='column' gap='size-50' flex={1}>
                    <Flex alignItems='center' gap='size-100'>
                        <Heading level={4} margin={0}>
                            {extension.name}
                        </Heading>
                        {extension.installed ? (
                            <Badge variant='positive'>v{extension.installed_version}</Badge>
                        ) : (
                            <Badge variant='neutral'>Available</Badge>
                        )}
                    </Flex>
                    <Text UNSAFE_className={classes.extensionDescription}>{extension.description}</Text>
                </Flex>
                {extension.installed ? (
                    <Button variant='secondary' isDisabled={isBusy} onPress={() => onUninstall(extension.id)}>
                        {isThisBusy ? 'Uninstalling…' : 'Uninstall'}
                    </Button>
                ) : (
                    <Button variant='secondary' isDisabled={isBusy} onPress={() => onInstall(extension.id)}>
                        {isThisBusy ? 'Installing…' : 'Install'}
                    </Button>
                )}
            </Flex>
        </View>
    );
};

const PluginCard = ({
    plugin,
    isBusy,
    busyId,
    onInstall,
    onUninstall,
}: {
    plugin: SchemaPluginResponse;
    isBusy: boolean;
    busyId: string | undefined;
    onInstall: (pluginId: string) => void;
    onUninstall: (pluginId: string) => void;
}) => {
    const isInstalled = plugin.installed;
    const isInUse = plugin.in_use_robot_count > 0;
    const isInstalling = busyId === plugin.id && isBusy;
    const extensions = plugin.extensions ?? [];

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
                {extensions.length > 0 ? (
                    isInstalled ? (
                        <Flex direction='column' gap='size-100'>
                            <Heading level={4}>Extensions</Heading>
                            {extensions.map((extension) => (
                                <ExtensionRow
                                    key={extension.id}
                                    extension={extension}
                                    isBusy={isBusy}
                                    busyId={busyId}
                                    onInstall={onInstall}
                                    onUninstall={onUninstall}
                                />
                            ))}
                        </Flex>
                    ) : (
                        <Text UNSAFE_className={classes.extensionHint}>
                            {extensions.length} extension{extensions.length === 1 ? '' : 's'} become available after
                            installing this plugin.
                        </Text>
                    )
                ) : null}
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
                        <Button
                            variant='secondary'
                            isDisabled={isInUse || isBusy}
                            onPress={() => onUninstall(plugin.id)}
                        >
                            Uninstall
                        </Button>
                    ) : (
                        <Button variant='primary' isDisabled={isBusy} onPress={() => onInstall(plugin.id)}>
                            {isInstalling ? 'Installing…' : 'Install'}
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
    const [busyId, setBusyId] = useState<string | undefined>(undefined);

    const plugins = pluginsQuery.data;
    const installed = plugins.filter((plugin) => plugin.installed);
    const available = plugins.filter((plugin) => !plugin.installed);
    const isBusy = busyId !== undefined;

    const install = async (pluginId: string) => {
        setBusyId(pluginId);
        try {
            await installMutation.mutateAsync({ params: { path: { plugin_id: pluginId } } });
            setRestartRequired(true);
            toast.positive('Plugin installed. Restart the server to activate it.');
        } catch (error) {
            toast.negative(getApiErrorMessage(error) ?? 'Failed to install the plugin.');
        } finally {
            setBusyId(undefined);
        }
    };

    const uninstall = async (pluginId: string) => {
        setBusyId(pluginId);
        try {
            await uninstallMutation.mutateAsync({ params: { path: { plugin_id: pluginId } } });
            setRestartRequired(true);
            toast.positive('Plugin uninstalled. Restart the server to apply the change.');
        } catch (error) {
            if (isResourceInUseError(error)) {
                toast.info(getApiErrorMessage(error) ?? 'This plugin is in use and cannot be uninstalled.');
                return;
            }
            toast.negative(getApiErrorMessage(error) ?? 'Failed to uninstall the plugin.');
        } finally {
            setBusyId(undefined);
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
                            <PluginCard
                                key={plugin.id}
                                plugin={plugin}
                                isBusy={isBusy}
                                busyId={busyId}
                                onInstall={install}
                                onUninstall={uninstall}
                            />
                        ))}
                    </Flex>
                </Flex>
            ) : null}
            {available.length > 0 ? (
                <Flex direction='column' gap='size-150'>
                    <Heading level={2}>Available</Heading>
                    <Flex gap='size-200' wrap>
                        {available.map((plugin) => (
                            <PluginCard
                                key={plugin.id}
                                plugin={plugin}
                                isBusy={isBusy}
                                busyId={busyId}
                                onInstall={install}
                                onUninstall={uninstall}
                            />
                        ))}
                    </Flex>
                </Flex>
            ) : null}
        </Flex>
    );
};
