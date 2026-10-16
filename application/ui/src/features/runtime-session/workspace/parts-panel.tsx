import { useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Heading,
    Item,
    Menu,
    MenuTrigger,
    Picker,
    ProgressCircle,
    StatusLight,
    Text,
    View,
} from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { SchemaEnvironmentDraft } from '../../../api/openapi-spec';
import { useProject, useProjectId } from '../../projects/use-project';
import { useRuntimeSession } from '../runtime-session-provider';
import { PendingConfirmation, SessionCommand, usePartCommand } from './use-part-command';

interface Choice {
    id: string;
    name: string;
}

/** Pick another environment or dataset to load in place of the current one. */
const SwapDialog = ({
    title,
    choices,
    current,
    onPick,
    close,
}: {
    title: string;
    choices: Choice[];
    current: string | null | undefined;
    onPick: (id: string) => void;
    close: () => void;
}) => {
    const [selected, setSelected] = useState<string | null>(current ?? null);

    return (
        <Dialog>
            <Heading>{title}</Heading>
            <Divider />
            <Content>
                <Picker
                    label={title}
                    width='100%'
                    selectedKey={selected}
                    onSelectionChange={(key) => setSelected(key === null ? null : String(key))}
                    items={choices}
                >
                    {(choice) => <Item key={choice.id}>{choice.name}</Item>}
                </Picker>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    isDisabled={selected === null}
                    onPress={() => {
                        close();
                        if (selected !== null) {
                            onPick(selected);
                        }
                    }}
                >
                    Load
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

/** Confirm a load that unloads parts which no longer fit, listing why. */
const ConfirmUnload = ({ confirmation }: { confirmation: PendingConfirmation }) => (
    <AlertDialog
        title='Unload parts that no longer fit?'
        variant='warning'
        primaryActionLabel='Unload and continue'
        secondaryActionLabel='Cancel'
        onPrimaryAction={confirmation.confirm}
        onSecondaryAction={confirmation.cancel}
    >
        <Flex direction='column' gap='size-100'>
            {confirmation.parts.map((part) => (
                <View key={`${part.part}-${part.name}`}>
                    <Text>
                        The {part.part} <strong>{part.name}</strong> will be unloaded:
                    </Text>
                    <ul>
                        {part.reasons.map((reason) => (
                            <li key={reason}>{reason}</li>
                        ))}
                    </ul>
                </View>
            ))}
        </Flex>
    </AlertDialog>
);

type PartAction = 'reload' | 'swap' | 'unload';

const PartRow = ({
    label,
    name,
    detail,
    isLoaded,
    isPending,
    actions,
    onAction,
}: {
    label: string;
    name: string | undefined;
    detail?: string;
    isLoaded: boolean;
    isPending: boolean;
    actions: PartAction[];
    onAction: (action: PartAction) => void;
}) => (
    <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
        <Flex direction='column' minWidth={0}>
            <StatusLight variant={isPending ? 'notice' : isLoaded ? 'positive' : 'neutral'}>{label}</StatusLight>
            <Text marginStart='size-300' UNSAFE_style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {isPending ? 'Loading…' : (name ?? 'Not loaded')}
                {detail && !isPending ? ` · ${detail}` : ''}
            </Text>
        </Flex>
        {isPending ? (
            <ProgressCircle size='S' isIndeterminate aria-label={`Loading ${label}`} />
        ) : (
            <MenuTrigger>
                <ActionButton isQuiet aria-label={`${label} actions`}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={(key) => onAction(key as PartAction)}>
                    {actions.map((action) => (
                        <Item key={action}>
                            {action === 'reload'
                                ? 'Reload'
                                : action === 'swap'
                                  ? isLoaded
                                      ? 'Swap…'
                                      : 'Load…'
                                  : 'Unload'}
                        </Item>
                    ))}
                </Menu>
            </MenuTrigger>
        )}
    </Flex>
);

/**
 * What the session has loaded, with reload, swap and unload.
 *
 * Swapping the environment can change the feature spec; the session then refuses, and the
 * confirmation lists what would be unloaded before the load is sent again.
 */
export const PartsPanel = () => {
    const { project_id } = useProjectId();
    const project = useProject();
    const { state } = useRuntimeSession();
    const { run, isPending, confirmation } = usePartCommand();
    const [pendingPart, setPendingPart] = useState<string | null>(null);
    const [swapping, setSwapping] = useState<'environment' | 'dataset' | null>(null);

    const { data: environments = [] } = $api.useQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });
    const datasets = project.datasets ?? [];

    const environment = state?.environment;
    const dataset = state?.dataset;

    const draftOf = (environment_id: string): SchemaEnvironmentDraft | undefined => {
        const saved = environments.find((candidate) => candidate.id === environment_id);
        return saved && { id: saved.id, name: saved.name, robots: saved.robots ?? [], cameras: saved.cameras ?? [] };
    };

    const runFor = async (part: string, command: SessionCommand) => {
        setPendingPart(part);
        try {
            await run(command);
        } finally {
            setPendingPart(null);
        }
    };

    const loadEnvironment = (environment_id: string) => {
        const draft = draftOf(environment_id);
        if (draft !== undefined) {
            void runFor('environment', { command: 'load_environment', environment: draft, unload_incompatible: false });
        }
    };

    const onEnvironment = (action: PartAction) => {
        if (action === 'swap') {
            setSwapping('environment');
        } else if (action === 'unload') {
            void runFor('environment', { command: 'unload_environment' });
        } else if (environment?.environment_id) {
            loadEnvironment(environment.environment_id);
        }
    };

    const onDataset = (action: PartAction) => {
        if (action === 'swap') {
            setSwapping('dataset');
        } else if (action === 'unload') {
            void runFor('dataset', { command: 'unload_dataset' });
        } else if (dataset) {
            // Loading the same dataset again finishes the recording (copies it back) and reopens it.
            void runFor('dataset', { command: 'load_dataset', dataset_id: dataset.dataset_id });
        }
    };

    const loaded = (isLoaded: boolean): PartAction[] => (isLoaded ? ['reload', 'swap', 'unload'] : ['swap']);

    return (
        <Flex direction='column' gap='size-150'>
            <Heading level={4} margin={0}>
                State
            </Heading>
            <PartRow
                label='Environment'
                name={environment?.name}
                isLoaded={environment != null}
                isPending={isPending && pendingPart === 'environment'}
                actions={environment?.environment_id ? loaded(true) : environment ? ['swap', 'unload'] : ['swap']}
                onAction={onEnvironment}
            />
            <PartRow
                label='Dataset'
                name={dataset?.name}
                detail={dataset?.episodes_recorded ? `${dataset.episodes_recorded} recorded this session` : undefined}
                isLoaded={dataset != null}
                isPending={isPending && pendingPart === 'dataset'}
                actions={environment == null ? [] : loaded(dataset != null)}
                onAction={onDataset}
            />

            <DialogContainer onDismiss={() => setSwapping(null)}>
                {swapping === 'environment' && (
                    <SwapDialog
                        title='Environment'
                        choices={environments.flatMap((choice) =>
                            choice.id ? [{ id: choice.id, name: choice.name }] : []
                        )}
                        current={environment?.environment_id}
                        onPick={loadEnvironment}
                        close={() => setSwapping(null)}
                    />
                )}
                {swapping === 'dataset' && (
                    <SwapDialog
                        title='Dataset'
                        choices={datasets.flatMap((choice) =>
                            choice.id ? [{ id: choice.id, name: choice.name }] : []
                        )}
                        current={dataset?.dataset_id}
                        onPick={(dataset_id) => void runFor('dataset', { command: 'load_dataset', dataset_id })}
                        close={() => setSwapping(null)}
                    />
                )}
            </DialogContainer>
            <DialogContainer onDismiss={() => confirmation?.cancel()}>
                {confirmation !== null && <ConfirmUnload confirmation={confirmation} />}
            </DialogContainer>
        </Flex>
    );
};
