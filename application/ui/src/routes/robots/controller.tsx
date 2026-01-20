import { Grid, View } from '@geti/ui';

import { JointControls } from '../../features/robots/controller/joint-controls';
import { RobotViewer } from '../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { useRobot } from '../../features/robots/use-robot';

export const Controller = () => {
    // NOTE: this route should be disabled if the robot hasn't been configured yet
    const robot = useRobot();
    return (
        <RobotModelsProvider>
            <Grid
                gap='size-200'
                UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-100)' }}
                areas={['controls controller', 'controls controller']}
                columns={['30rem', '1fr']}
                //rows={['min-content', 'auto']}
                height='100%'
                minHeight={0}
            >
                <View gridArea='controller'>
                    <RobotViewer robot={robot} />
                </View>
                <JointControls />
            </Grid>
        </RobotModelsProvider>
    );
};
