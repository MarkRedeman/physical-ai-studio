import { TextField } from '@geti/ui';

import { SchemaModel } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';
import { SchemaTrainJob, TrainModelDialog } from './train-model-dialog';

export const RetrainModelModal = ({
    baseModel,
    close,
}: {
    baseModel: SchemaModel;
    close: (job: SchemaTrainJob | undefined) => void;
}) => {
    const { datasets, id: project_id } = useProject();

    return (
        <TrainModelDialog
            title='Retrain Model'
            submitLabel='Retrain'
            datasets={datasets}
            projectId={project_id}
            defaultName={baseModel.name}
            defaultDatasetId={baseModel.dataset_id}
            defaultMaxSteps={10000}
            extraPayload={{ base_model_id: baseModel.id! }}
            getPolicy={() => baseModel.policy}
            close={close}
            policyField={<TextField label='Policy' value={baseModel.policy.toUpperCase()} isReadOnly />}
        />
    );
};
