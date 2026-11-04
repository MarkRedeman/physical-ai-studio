import { useState } from 'react';

import { ActionButton, Content, Dialog, DialogTrigger, Flex, Heading, SearchField, Text, TextArea } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaEpisodeInfo } from '../../../api/openapi-spec';

import classes from './current-task-picker.module.css';

export const getTaskSuggestions = (episodes: SchemaEpisodeInfo[], recentTasks: string[]): string[] => {
    const recordedTasks = episodes
        .toSorted((a, b) => b.episode_index - a.episode_index)
        .flatMap((episode) => episode.tasks);
    return [...new Set([...recentTasks, ...recordedTasks].map((task) => task.trim()).filter(Boolean))];
};

const TaskHistory = ({
    tasks,
    onSelect,
    close,
}: {
    tasks: string[];
    onSelect: (task: string) => void;
    close: () => void;
}) => {
    const [search, setSearch] = useState('');
    const matching = tasks.filter((task) => task.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()));

    return (
        <Dialog>
            <Heading>Previous tasks</Heading>
            <Content>
                <Flex direction='column' gap='size-150'>
                    <SearchField label='Find a task' width='100%' value={search} onChange={setSearch} autoFocus />
                    {matching.length === 0 ? (
                        <Text>No matching tasks. Try a different search.</Text>
                    ) : (
                        <ul className={classes.taskList} aria-label='Previous tasks'>
                            {matching.map((task) => (
                                <li key={task}>
                                    <button
                                        type='button'
                                        className={classes.taskOption}
                                        onClick={() => {
                                            onSelect(task);
                                            close();
                                        }}
                                    >
                                        {task}
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </Flex>
            </Content>
        </Dialog>
    );
};

export const CurrentTaskPicker = ({
    dataset_id,
    task,
    onChange,
    recentTasks,
    isReadOnly,
}: {
    dataset_id: string;
    task: string;
    onChange: (task: string) => void;
    recentTasks: string[];
    isReadOnly: boolean;
}) => {
    const { data: episodes = [], isError } = $api.useQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: { path: { dataset_id } },
    });
    const suggestions = getTaskSuggestions(episodes, recentTasks);

    return (
        <Flex direction='column' gap='size-50'>
            <TextArea
                label='Current task'
                width='100%'
                isRequired
                isReadOnly={isReadOnly}
                value={task}
                onChange={onChange}
                description='Describe the action to capture in this episode.'
            />
            {suggestions.length > 0 && (
                <DialogTrigger type='popover'>
                    <ActionButton isQuiet isDisabled={isReadOnly} alignSelf='start'>
                        Use previous task…
                    </ActionButton>
                    {(close) => <TaskHistory tasks={suggestions} onSelect={onChange} close={close} />}
                </DialogTrigger>
            )}
            {isError && (
                <Text UNSAFE_className={classes.hint}>
                    Previous tasks could not be loaded. You can still enter a new task.
                </Text>
            )}
        </Flex>
    );
};
