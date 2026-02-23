import { Outlet } from 'react-router';

import { RobotFormProvider } from '../../features/robots/robot-form/provider';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { TrossenDebugProvider } from '../../features/robots/trossen/setup-wizard/debug-panel';

/**
 * Shared layout for the "Add new robot" flow.
 *
 * Wraps child routes with RobotModelsProvider and RobotFormProvider so that
 * form state (name, type, serial_number, connection_string) is preserved when
 * navigating between the generic form (/robots/new) and the setup wizards
 * (/robots/new/so101-setup, /robots/new/trossen-setup).
 *
 * TrossenDebugProvider sits at this level so the floating debug panel is
 * visible on every child route (including the robot-type selection page),
 * not only inside the Trossen setup wizard.
 */
export const NewRobotLayout = () => {
    return (
        <RobotModelsProvider>
            <RobotFormProvider>
                <TrossenDebugProvider>
                    <Outlet />
                </TrossenDebugProvider>
            </RobotFormProvider>
        </RobotModelsProvider>
    );
};
