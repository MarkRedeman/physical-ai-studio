import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Checkbox,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    Link,
    SearchField,
    Text,
    View,
} from '@geti-ui/ui';
import { clsx } from 'clsx';

import { ReactComponent as PhysicalAIStudioLogo } from '../../../assets/icons/physicalai-studio-logo.svg';
import so101BimanualThumbnail from '../../../assets/thumbnails/BimanualSO101_Follower_thumbnail.png';
import leKiwiThumbnail from '../../../assets/thumbnails/LeKiwi_Follower_thumbnail.png';
import leRobotThumbnail from '../../../assets/thumbnails/LeRobot_thumbnail.png';
import mujocoThumbnail from '../../../assets/thumbnails/MuJoCo_thumbnail.png';
import reBotArm102Thumbnail from '../../../assets/thumbnails/ReBot_Arm102_Leader_thumbnail.png';
import reBotB601Thumbnail from '../../../assets/thumbnails/ReBot_B601_DM_Follower_thumbnail.png';
import so101Thumbnail from '../../../assets/thumbnails/SO101_Leader_thumbnail.png';
import trossenBimanualThumbnail from '../../../assets/thumbnails/Trossen_Bimanual_WidowXAI_Follower_thumbnail.png';
import trossenThumbnail from '../../../assets/thumbnails/Trossen_WidowXAI_Follower_thumbnail.png';
import { useRobotCatalogQuery } from '../robot-catalog.hooks';
import { useRobotForm } from './provider';

import classes from './robot-catalog-dialog.module.css';

type RobotRoleFilter = 'all' | 'follower' | 'leader';

const ROLE_CLASS_NAMES = {
    follower: classes.follower,
    leader: classes.leader,
} as const;

type CatalogManifest = {
    plugin_category: string;
    description: string;
    github_link: string;
    thumbnails?: Record<string, string>;
};

type CatalogEntry = ReturnType<typeof useRobotCatalogQuery>['data'][number];

export const CATALOG_MANIFEST: Record<string, CatalogManifest> = {
    SO101: {
        plugin_category: 'SO101',
        description: 'LeRobot SO-101 arms for learning from demonstration.',
        github_link: 'https://github.com/huggingface/lerobot',
        thumbnails: {
            SO101_Follower: so101Thumbnail,
            SO101_Leader: so101Thumbnail,
            BimanualSO101_Follower: so101BimanualThumbnail,
            BimanualSO101_Leader: so101BimanualThumbnail,
        },
    },
    Trossen: {
        plugin_category: 'Trossen',
        description: 'Trossen Robotics WidowX AI arm integrations.',
        github_link: 'https://github.com/TrossenRobotics',
        thumbnails: {
            Trossen_WidowXAI_Follower: trossenThumbnail,
            Trossen_WidowXAI_Leader: trossenThumbnail,
            Trossen_Bimanual_WidowXAI_Follower: trossenBimanualThumbnail,
            Trossen_Bimanual_WidowXAI_Leader: trossenBimanualThumbnail,
        },
    },
    ReBot: {
        plugin_category: 'ReBot',
        description: 'ReBot B601 and Arm102 robot integrations.',
        github_link: 'https://github.com/open-edge-platform/physical-ai-rebot-b601-plugin',
        thumbnails: {
            ReBot_B601_DM_Follower: reBotB601Thumbnail,
            ReBot_Arm102_Leader: reBotArm102Thumbnail,
        },
    },
    LeRobot: {
        plugin_category: 'LeRobot',
        description: 'Robot and teleoperator configurations discovered from LeRobot.',
        github_link: 'https://github.com/huggingface/lerobot',
    },
    LeKiwi: {
        plugin_category: 'LeKiwi',
        description: 'LeKiwi mobile manipulator integration.',
        github_link: 'https://github.com/huggingface/lerobot',
        thumbnails: {
            LeKiwi_Follower: leKiwiThumbnail,
            LeKiwi_Leader: leKiwiThumbnail,
        },
    },
    MuJoCo: {
        plugin_category: 'MuJoCo',
        description: 'MuJoCo-backed SO-101 simulation integration.',
        github_link: 'https://github.com/google-deepmind/mujoco',
        thumbnails: {
            MuJoCo_SO101_Follower: mujocoThumbnail,
        },
    },
};

const RobotCard = ({
    entry,
    category,
    activeType,
    onSelect,
}: {
    entry: CatalogEntry;
    category: string;
    activeType: string | undefined;
    onSelect: () => void;
}) => {
    const thumbnail =
        category === 'LeRobot'
            ? leRobotThumbnail
            : (CATALOG_MANIFEST[category]?.thumbnails?.[entry.type] ?? entry.preview_thumbnail);

    return (
        <Button
            variant={activeType === entry.type ? 'accent' : 'secondary'}
            onPress={onSelect}
            UNSAFE_className={classes.card}
            UNSAFE_style={{ alignItems: 'flex-start', justifyContent: 'flex-start' }}
        >
            <div className={classes.cardContent}>
                <div className={classes.thumbnailArea}>
                    {thumbnail ? (
                        <img className={classes.thumbnail} src={thumbnail} alt='' />
                    ) : (
                        <div className={classes.thumbnailFallback}>
                            <PhysicalAIStudioLogo width={56} height={56} style={{ filter: 'grayscale(100%)' }} />
                        </div>
                    )}
                </div>
                <span aria-label={entry.role} className={clsx(classes.role, ROLE_CLASS_NAMES[entry.role])} />
                <div className={classes.cardDetails}>
                    <Text>{entry.display_name}</Text>
                </div>
            </div>
        </Button>
    );
};

export const RobotCatalogDialog = ({ close }: { close: () => void }) => {
    const { activeType, setActiveType } = useRobotForm();
    const catalog = useRobotCatalogQuery();
    const [role, setRole] = useState<RobotRoleFilter>('all');
    const [showExternal, setShowExternal] = useState(true);
    const [search, setSearch] = useState('');
    const normalizedSearch = search.trim().toLocaleLowerCase();
    const entries = catalog.data.filter(
        (entry) =>
            (role === 'all' || entry.role === role) &&
            (showExternal || entry.source !== 'external') &&
            (normalizedSearch === '' ||
                entry.display_name.toLocaleLowerCase().includes(normalizedSearch) ||
                entry.category.toLocaleLowerCase().includes(normalizedSearch))
    );
    const categories = new Map<string, typeof entries>();
    entries.forEach((entry) => {
        // if (entry.category === 'MuJoCo') {
        //     return;
        // }
        categories.set(entry.category, [...(categories.get(entry.category) ?? []), entry]);
    });

    const selectRobot = (type: string) => {
        setActiveType(type);
        close();
    };

    return (
        <Dialog size='L' width={'100%'} height='100%'>
            <Heading>
                <Flex width='100%' justifyContent={'space-between'}>
                    <span>Select robot type</span>
                    <Flex alignItems={'center'} gap='size-200'>
                        <ButtonGroup aria-label='Robot role filter'>
                            {(['all', 'follower', 'leader'] as const).map((filter) => (
                                <Button
                                    key={filter}
                                    variant={role === filter ? 'accent' : 'secondary'}
                                    onPress={() => setRole(filter)}
                                    UNSAFE_className={
                                        filter === 'all'
                                            ? undefined
                                            : clsx(classes.filterRole, ROLE_CLASS_NAMES[filter])
                                    }
                                >
                                    {filter === 'all' ? 'All roles' : `${filter[0].toUpperCase()}${filter.slice(1)}s`}
                                </Button>
                            ))}
                        </ButtonGroup>
                        <Checkbox isSelected={showExternal} onChange={setShowExternal}>
                            Show external plugins
                        </Checkbox>
                        <SearchField
                            aria-label='Search robot types'
                            placeholder='Search robots'
                            value={search}
                            onChange={setSearch}
                            onClear={() => setSearch('')}
                            width='size-3600'
                        />
                    </Flex>
                </Flex>
            </Heading>
            <Divider />
            <Content UNSAFE_className={classes.content}>
                {[...categories].map(([category, robots]) => (
                    <View key={category}>
                        <Flex alignItems='baseline' gap='size-150'>
                            <Heading level={3}>{category}</Heading>
                            <Text>{CATALOG_MANIFEST[category]?.description ?? 'Robot integration plugin.'}</Text>
                            {CATALOG_MANIFEST[category] && (
                                <Link href={CATALOG_MANIFEST[category].github_link} target='_blank'>
                                    GitHub
                                </Link>
                            )}
                        </Flex>
                        <Flex gap='size-200' UNSAFE_className={classes.robotRow}>
                            {robots.map((entry) => (
                                <RobotCard
                                    key={entry.type}
                                    entry={entry}
                                    category={category}
                                    activeType={activeType}
                                    onSelect={() => selectRobot(entry.type)}
                                />
                            ))}
                        </Flex>
                    </View>
                ))}
                {entries.length === 0 && <Text>No robots match the selected filters.</Text>}
            </Content>
        </Dialog>
    );
};
