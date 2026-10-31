import { FormPreviewLayout } from '../../components/form-preview-layout';
import { CreateRobotForm } from '../../features/robots/robot-form/create-form';
import { LiveConnectProvider } from '../../features/robots/robot-form/live-connect/provider';
import { Preview } from '../../features/robots/robot-form/preview';

export const New = () => {
    return (
        <LiveConnectProvider>
            <FormPreviewLayout form={<CreateRobotForm />} preview={<Preview />} />
        </LiveConnectProvider>
    );
};
