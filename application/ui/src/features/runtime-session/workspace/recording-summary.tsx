import { Content, ContextualHelp, Flex, Heading, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { useRuntimeSession } from '../runtime-session-provider';

import classes from './record-mode.module.css';

export const RecordingSummary = () => {
    const { state } = useRuntimeSession();
    const dataset = state?.dataset;
    const { data: episodes } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/episodes',
        {
            params: { path: { dataset_id: dataset?.dataset_id ?? '' } },
        },
        { enabled: dataset != null }
    );

    if (dataset == null || episodes === undefined) return null;

    // Session episodes are copied into the dataset when recording finishes.
    const total = episodes.length + dataset.episodes_recorded;
    return (
        <Flex alignItems='center' gap='size-50'>
            <Text UNSAFE_className={classes.hint}>
                {total} {total === 1 ? 'episode' : 'episodes'} in dataset
            </Text>
            <ContextualHelp variant='info'>
                <Heading>Building your dataset</Heading>
                <Content>
                    <Text>
                        We recommend recording at least 50 episodes before training your first model. This total
                        includes episodes saved in this session.
                    </Text>
                </Content>
            </ContextualHelp>
        </Flex>
    );
};
