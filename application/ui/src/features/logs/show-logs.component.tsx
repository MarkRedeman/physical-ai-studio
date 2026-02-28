// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useEffect, useMemo, useState } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Icon,
    Item,
    Loading,
    Picker,
    Section,
    Text,
    View,
} from '@geti/ui';
import { LogsIcon } from '@geti/ui/icons';
import { queryOptions, experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchSSE } from '../../api/fetch-sse';
import type { LogEntry, LogSource } from './log-types';
import { LogViewer } from './log-viewer.component';

import styles from './log-viewer.module.scss';

// ---- Sources query ----

const useLogSources = () => {
    return useQuery<LogSource[]>({
        queryKey: ['get', '/api/logs/sources'],
        queryFn: async () => {
            const res = await fetch('/api/logs/sources');
            if (!res.ok) throw new Error('Failed to fetch log sources');
            return res.json();
        },
        staleTime: 30_000,
    });
};

// ---- SSE log stream ----

const LogStreamViewer = ({ sourceId }: { sourceId: string }) => {
    const query = useQuery(
        queryOptions({
            queryKey: ['get', '/api/logs/{source_id}/stream', sourceId],
            queryFn: streamedQuery({
                queryFn: () => fetchSSE<LogEntry>(`/api/logs/${sourceId}/stream`),
            }),
            staleTime: Infinity,
        })
    );

    const validLogs = useMemo(() => {
        if (!query.data) return [];

        return query.data.filter((entry): entry is LogEntry => {
            return (
                entry !== null &&
                typeof entry === 'object' &&
                'record' in entry &&
                entry.record !== null &&
                typeof entry.record === 'object' &&
                'level' in entry.record &&
                'time' in entry.record &&
                'message' in entry.record
            );
        });
    }, [query.data]);

    return <LogViewer logs={validLogs} isLoading={query.isLoading} />;
};

// ---- Dialog content ----

const LogsDialogContent = ({ close }: { close: () => void }) => {
    const { data: sources, isLoading: sourcesLoading } = useLogSources();
    const [selectedSourceId, setSelectedSourceId] = useState<string>('app');

    // Reset to 'app' if the current selection is no longer available
    useEffect(() => {
        if (sources && sources.length > 0) {
            const exists = sources.some((s) => s.id === selectedSourceId);
            if (!exists) {
                setSelectedSourceId(sources[0].id);
            }
        }
    }, [sources, selectedSourceId]);

    const workerSources = useMemo(() => (sources ?? []).filter((s) => s.type === 'worker'), [sources]);
    const jobSources = useMemo(() => (sources ?? []).filter((s) => s.type === 'job'), [sources]);

    return (
        <Dialog>
            <Heading>Logs</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200' height='100%'>
                    <View UNSAFE_className={styles.sourcePickerContainer}>
                        <Flex alignItems='center' gap='size-200'>
                            <Text>Source:</Text>
                            <Picker
                                aria-label='Log source'
                                selectedKey={selectedSourceId}
                                onSelectionChange={(key) => setSelectedSourceId(String(key))}
                                isDisabled={sourcesLoading}
                                width='size-3000'
                            >
                                {workerSources.length > 0 ? (
                                    <Section title='Workers'>
                                        {workerSources.map((s) => (
                                            <Item key={s.id}>{s.name}</Item>
                                        ))}
                                    </Section>
                                ) : (
                                    <></>
                                )}
                                {jobSources.length > 0 ? (
                                    <Section title='Jobs'>
                                        {jobSources.map((s) => (
                                            <Item key={s.id}>{s.name}</Item>
                                        ))}
                                    </Section>
                                ) : (
                                    <></>
                                )}
                            </Picker>
                        </Flex>
                    </View>

                    <Suspense fallback={<Loading mode='inline' />}>
                        <LogStreamViewer sourceId={selectedSourceId} />
                    </Suspense>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

// ---- Exported trigger button ----

export const ShowLogs = () => {
    return (
        <DialogTrigger type='fullscreen'>
            <ActionButton aria-label='View logs'>
                <Icon>
                    <LogsIcon />
                </Icon>
                <Text>Logs</Text>
            </ActionButton>
            {(close) => <LogsDialogContent close={close} />}
        </DialogTrigger>
    );
};
