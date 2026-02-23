import { View } from '@geti/ui';

import { TrossenSetupWizardContent } from '../../features/robots/trossen/setup-wizard/setup-wizard';
import { TrossenSetupWizardProvider } from '../../features/robots/trossen/setup-wizard/wizard-provider';

/**
 * Route: /projects/:project_id/robots/new/trossen-setup
 *
 * Dedicated route for the Trossen WidowX AI setup wizard. Rendered as a child
 * of NewRobotLayout which provides RobotFormProvider and RobotModelsProvider,
 * so form state (name, type, connection_string) is shared with the generic
 * form and preserved across navigation.
 */
export const TrossenSetup = () => {
    return (
        <TrossenSetupWizardProvider>
            <View height='100%' backgroundColor='gray-100' padding='size-400' UNSAFE_style={{ overflow: 'hidden' }}>
                <TrossenSetupWizardContent />
            </View>
        </TrossenSetupWizardProvider>
    );
};
