import type { ReactNode } from 'react';

import { ActionButton, Icon } from '@geti-ui/ui';
import { Refresh } from '@geti-ui/ui/icons';

import { useCatalogIdentifyMutation, useDiscoverRobotsQuery } from '../../robot-catalog.hooks';
import type { SchemaRobotType } from '../../robot-types';

import classes from '../form.module.css';

export const RefreshRobotsButton = ({ robotType }: { robotType: SchemaRobotType }) => {
    const { refetch, isFetching } = useDiscoverRobotsQuery(robotType);

    return (
        <ActionButton
            isDisabled={isFetching}
            UNSAFE_className={classes.actionButton}
            onPress={() => {
                refetch();
            }}
        >
            <Icon>
                <Refresh />
            </Icon>
        </ActionButton>
    );
};

export const IdentifyRobot = ({
    robotType,
    payload,
    errorElement,
}: {
    robotType: SchemaRobotType;
    payload: Record<string, unknown> | null;
    errorElement?: ReactNode;
}) => {
    const identifyMutation = useCatalogIdentifyMutation();
    const isDisabled = payload === null || identifyMutation.isPending;

    const onIdentify = () => {
        if (isDisabled || payload === null) {
            return;
        }

        identifyMutation.mutate({
            params: { path: { robot_type: robotType } },
            body: payload,
        });
    };

    return (
        <>
            <ActionButton isDisabled={isDisabled} UNSAFE_className={classes.actionButton} onPress={onIdentify}>
                Identify
            </ActionButton>
            {identifyMutation.isError && errorElement}
        </>
    );
};
