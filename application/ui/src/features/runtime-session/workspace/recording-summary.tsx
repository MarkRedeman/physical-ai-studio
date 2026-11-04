import { ActionButton, Content, Dialog, DialogTrigger, Flex, Heading, Icon, Link, Text } from '@geti-ui/ui';
import { InfoOutline } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { useRuntimeSession } from '../runtime-session-provider';

import classes from './recording-summary.module.css';

const RECOMMENDED_EPISODES = 50;

const RecordingTips = () => (
    <Dialog>
        <Heading>Recording a useful dataset</Heading>
        <Content>
            <Flex direction='column' gap='size-150'>
                <Text>
                    Aim for 50+ episodes before your first training run. This is a starting recommendation, not a
                    requirement.
                </Text>
                <ul className={classes.tips}>
                    <li>
                        <strong>Check camera views.</strong> Keep the robot's actions and task objects clearly visible.
                    </li>
                    <li>
                        <strong>Capture variation.</strong> Keep the task consistent, but vary starting positions and
                        object placement.
                    </li>
                    <li>
                        <strong>Keep good demonstrations.</strong> Save successful attempts; discard incomplete or
                        unintended movements.
                    </li>
                    <li>
                        <strong>Finish the session.</strong> Use Back to dataset when you're done to finish saving your
                        episodes.
                    </li>
                </ul>
                <Link href='https://github.com/open-edge-platform/physical-ai-studio/issues/358'>
                    More about recommended dataset sizes
                </Link>
            </Flex>
        </Content>
    </Dialog>
);

/** One home for episode totals, this session's contribution, and dataset-building guidance. */
export const RecordingSummary = () => {
    const { state } = useRuntimeSession();
    const dataset = state?.dataset;
    const { data: episodes, isError } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/episodes',
        {
            params: { path: { dataset_id: dataset?.dataset_id ?? '' } },
        },
        { enabled: dataset != null }
    );

    if (dataset == null) return null;

    // Session episodes are copied into the dataset when recording finishes.
    const total = episodes === undefined ? undefined : episodes.length + dataset.episodes_recorded;
    const progress = total === undefined ? undefined : Math.min(total, RECOMMENDED_EPISODES);
    const isAtStartingPoint = total !== undefined && total >= RECOMMENDED_EPISODES;

    return (
        <div className={classes.summary} role='region' aria-label='Dataset progress'>
            <div className={classes.stats}>
                <div
                    className={classes.total}
                    role='group'
                    aria-label={
                        total === undefined
                            ? 'Counting episodes'
                            : `${total} ${total === 1 ? 'episode' : 'episodes'} in dataset`
                    }
                >
                    <strong className={classes.count}>{total ?? '—'}</strong>
                    <span className={classes.countLabel}>{total === 1 ? 'episode' : 'episodes'} in dataset</span>
                </div>
                {dataset.episodes_recorded > 0 && (
                    <span className={classes.sessionCount}>+{dataset.episodes_recorded} this session</span>
                )}
            </div>
            {progress !== undefined && (
                <div
                    className={classes.progress}
                    role='progressbar'
                    aria-label='Progress toward 50 recommended episodes'
                    aria-valuemin={0}
                    aria-valuemax={RECOMMENDED_EPISODES}
                    aria-valuenow={progress}
                    aria-valuetext={`${total} episodes collected; 50 recommended to start`}
                >
                    <span style={{ width: `${(progress / RECOMMENDED_EPISODES) * 100}%` }} />
                </div>
            )}
            <span className={classes.guidance}>
                {isError
                    ? 'Episode total unavailable. You can still record.'
                    : isAtStartingPoint
                      ? 'A useful starting point. Keep adding variety.'
                      : 'Aim for 50+ varied episodes before your first training run.'}
            </span>
            <DialogTrigger type='popover'>
                <ActionButton isQuiet alignSelf='start' UNSAFE_className={classes.helpButton}>
                    <Icon>
                        <InfoOutline />
                    </Icon>
                    <Text>Recording tips</Text>
                </ActionButton>
                <RecordingTips />
            </DialogTrigger>
        </div>
    );
};
