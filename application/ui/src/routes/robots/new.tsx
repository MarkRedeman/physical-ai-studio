import { Suspense } from 'react';

import { Button, Flex, Grid, Loading, minmax, View } from '@geti/ui';
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

/**
 * Submit button that adapts to the selected robot type:
 * - SO101 types: navigates to the SO101 setup wizard
 * - Trossen types: navigates to the Trossen setup wizard
 * - Other types: directly creates the robot via POST (default behavior)
 */
const NewRobotSubmitButton = () => {
    const robotForm = useRobotForm();
    const navigate = useNavigate();
    const { project_id } = useProjectId();

    const isSO101 = robotForm.type?.toLowerCase().startsWith('so101') ?? false;
    const isTrossen = robotForm.type?.toLowerCase().includes('trossen') ?? false;

    if (!isSO101 && !isTrossen) {
        return <SubmitNewRobotButton />;
    }

    if (isSO101) {
        const isDisabled = !robotForm.name || !robotForm.type || !robotForm.serial_number;

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
    }

    // Trossen
    const isDisabled = !robotForm.name || !robotForm.type || !robotForm.connection_string;

    return (
        <Button
            variant='accent'
            isDisabled={isDisabled}
            onPress={() => {
                navigate(paths.project.robots.trossenSetup({ project_id }));
            }}
        >
            Begin Setup
        </Button>
    );
};

export const New = () => {
    return (
        <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
            <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                <Suspense fallback={<CenteredLoading />}>
                    <RobotForm submitButton={<NewRobotSubmitButton />} />
                </Suspense>
            </View>
            <View gridArea='controls' backgroundColor={'gray-50'} padding='size-400'>
                <Preview />
            </View>
        </Grid>
    );
};
