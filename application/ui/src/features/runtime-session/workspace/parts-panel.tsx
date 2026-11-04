import { useEffect, useState } from 'react';

import {
    ActionButton,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Item,
    Menu,
    MenuTrigger,
    ProgressCircle,
    StatusLight,
    Text,
} from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { SchemaSessionEnvironment, SchemaStatusMessage } from '../../../api/openapi-spec';
import { failureOf, SessionPartName, Statuses } from '../runtime-session-provider';
import { PartAction } from './part-controls';

/** Card-header actions use task language; recovery appears only when something failed. */
export const PartActions = ({
    part,
    isLoaded,
    isPending,
    isDisabled,
    canChoose,
    canRetry,
    onAction,
}: {
    part: SessionPartName;
    isLoaded: boolean;
    isPending: boolean;
    isDisabled: boolean;
    canChoose: boolean;
    canRetry: boolean;
    onAction: (action: PartAction) => void;
}) => {
    if (isPending) return <ProgressCircle size='S' isIndeterminate aria-label={`Preparing ${part}`} />;
    if (!isLoaded)
        return (
            <ActionButton isQuiet isDisabled={isDisabled || !canChoose} onPress={() => onAction('swap')}>
                Choose {part}…
            </ActionButton>
        );

    const actions: { id: PartAction; label: string }[] = [{ id: 'swap', label: `Change ${part}…` }];
    if (canRetry) actions.push({ id: 'reload', label: part === 'environment' ? 'Reconnect devices' : 'Try again' });

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label={`${part} actions`} isDisabled={isDisabled || (!canChoose && !canRetry)}>
                <MoreMenu />
            </ActionButton>
            <Menu
                items={actions}
                disabledKeys={canChoose ? [] : ['swap']}
                onAction={(key) => onAction(key as PartAction)}
            >
                {(action) => <Item key={action.id}>{action.label}</Item>}
            </Menu>
        </MenuTrigger>
    );
};

/** Each of the environment's devices, as it last reported. */
const DeviceStatuses = ({
    devices,
    environment,
}: {
    devices: SchemaStatusMessage[];
    environment: SchemaSessionEnvironment;
}) => {
    const label = (device: string) =>
        ({ follower: 'Follower', leader: 'Leader' })[device] ??
        environment.cameras.find((camera) => camera.key === device)?.name ??
        device;

    return (
        <Flex direction='column' marginStart='size-300'>
            {devices.map(({ component, state, error }) => (
                <Flex key={component} direction='column'>
                    {state === 'failed' ? (
                        <StatusLight variant='negative'>{label(component.slice('environment/'.length))}</StatusLight>
                    ) : (
                        <Text>
                            {label(component.slice('environment/'.length))}
                            {state === 'starting' ? ' · Connecting…' : ''}
                        </Text>
                    )}
                    {error && state === 'failed' && (
                        <Text marginStart='size-300' UNSAFE_style={{ color: 'var(--spectrum-negative-visual-color)' }}>
                            {error}
                        </Text>
                    )}
                </Flex>
            ))}
        </Flex>
    );
};

/** A selected resource and its status; actions belong in its card header. */
export const PartRow = ({
    label,
    name,
    detail,
    error,
    isPending,
}: {
    label: string;
    name: string | undefined;
    detail?: string;
    error?: string | null;
    isPending: boolean;
}) => (
    <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
        <Flex direction='column' minWidth={0}>
            <Text UNSAFE_style={{ fontWeight: 500, overflowWrap: 'anywhere' }}>
                {isPending ? 'Preparing…' : (name ?? `No ${label.toLowerCase()} selected`)}
            </Text>
            {detail && !isPending && <Text>{detail}</Text>}
            {error && !isPending && (
                <Disclosure isQuiet defaultExpanded>
                    <DisclosureTitle>Error details</DisclosureTitle>
                    <DisclosurePanel>
                        <Text
                            UNSAFE_style={{ color: 'var(--spectrum-negative-visual-color)', overflowWrap: 'anywhere' }}
                        >
                            {error}
                        </Text>
                    </DisclosurePanel>
                </Disclosure>
            )}
        </Flex>
    </Flex>
);

/** Expand the environment's status to see individual devices and their errors. */
export const EnvironmentRow = ({
    environment,
    statuses,
    isPending,
}: {
    environment: SchemaSessionEnvironment | null | undefined;
    statuses: Statuses;
    isPending: boolean;
}) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const devices = Object.values(statuses).filter(({ component }) => component.startsWith('environment/'));
    const failed = devices.filter(({ state }) => state === 'failed').length;
    const failure = failureOf(statuses, 'environment');
    const canExpand = environment != null && (devices.length > 0 || failure !== null);
    const failures = devices
        .filter(({ state }) => state === 'failed')
        .map(({ component, error }) => `${component}:${error}`)
        .join('|');
    useEffect(() => {
        if (failure !== null || failures !== '') setIsExpanded(true);
    }, [failure, failures]);

    return (
        <>
            <PartRow
                label='Environment'
                name={environment?.name}
                error={canExpand ? undefined : failure}
                isPending={isPending}
            />
            {canExpand && (
                <Disclosure isQuiet isExpanded={isExpanded} onExpandedChange={setIsExpanded}>
                    <DisclosureTitle>
                        {failed > 0
                            ? `${failed} device${failed === 1 ? ' needs' : 's need'} attention`
                            : 'Device details'}
                    </DisclosureTitle>
                    <DisclosurePanel>
                        <Flex direction='column' gap='size-100'>
                            {failure !== null && (
                                <Text UNSAFE_style={{ color: 'var(--spectrum-negative-visual-color)' }}>{failure}</Text>
                            )}
                            <DeviceStatuses devices={devices} environment={environment} />
                        </Flex>
                    </DisclosurePanel>
                </Disclosure>
            )}
        </>
    );
};
