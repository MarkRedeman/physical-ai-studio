import { Suspense } from 'react';

import { Button, Flex, Grid, Loading, minmax, View } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { useProjectId } from '../../features/projects/use-project';
import { RobotForm } from '../../features/robots/robot-form/form';
import { Preview } from '../../features/robots/robot-form/preview';
import { useRobotForm } from '../../features/robots/robot-form/provider';
import { SubmitNewRobotButton } from '../../features/robots/robot-form/submit-new-robot-button';
import { paths } from '../../router';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

const BeginSO101SetupButton = ({ activeType }: { activeType: 'SO101_Follower' | 'SO101_Leader' }) => {
    const { robotForm } = useRobotForm(activeType);
    const navigate = useNavigate();
    const { project_id } = useProjectId();

    const isDisabled = !robotForm.name || !activeType || (!robotForm.serial_number && !robotForm.connection_string);

    return (
        <Button
            variant='accent'
            isDisabled={isDisabled}
            onPress={() => {
                navigate(paths.project.robots.so101Setup({ project_id }));
            }}
        >
            Begin Setup
        </Button>
    );
};

/**
 * Submit button that adapts to the selected robot type:
 * - SO101 types: navigates to the setup wizard (sibling route under same layout)
 * - Other types (Trossen): directly creates the robot via POST (default behavior)
 */
const NewRobotSubmitButton = () => {
    const { activeType } = useRobotForm();

    const isSO101 = activeType === 'SO101_Follower' || activeType === 'SO101_Leader';

    if (isSO101) {
        return <BeginSO101SetupButton activeType={activeType} />;
    }

    return <SubmitNewRobotButton />;
};

export const New = () => {
    return (
        <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
            <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                <Suspense fallback={<CenteredLoading />}>
                    <RobotForm submitButton={<NewRobotSubmitButton />} />
                </Suspense>
            </View>
            <View gridArea='controls'>
                <Preview />
            </View>
        </Grid>
    );
};
