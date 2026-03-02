import { useState } from 'react';

import { Item, Key, Picker } from '@geti/ui';

import { useProject } from '../../features/projects/use-project';
import { SchemaTrainJob, TrainModelDialog } from './train-model-dialog';

export type { SchemaTrainJob };

export const TrainModelModal = (close: (job: SchemaTrainJob | undefined) => void) => {
    const { datasets, id: project_id } = useProject();
    const [selectedPolicy, setSelectedPolicy] = useState<Key | null>('act');

    return (
        <TrainModelDialog
            title='Train Model'
            submitLabel='Train'
            datasets={datasets}
            projectId={project_id}
            getPolicy={() => selectedPolicy?.toString()}
            close={close}
            policyField={
                <Picker label='Policy' selectedKey={selectedPolicy} onSelectionChange={setSelectedPolicy}>
                    <Item key='act'>Act</Item>
                    <Item key='pi0'>Pi0</Item>
                    <Item key='smolvla'>SmolVLA</Item>
                </Picker>
            }
        />
    );
};
