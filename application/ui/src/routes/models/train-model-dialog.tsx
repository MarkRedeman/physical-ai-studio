import { ReactNode, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Divider,
    Flex,
    Form,
    Heading,
    Item,
    Key,
    NumberField,
    Picker,
    TextField,
} from '@geti/ui';

import { $api } from '../../api/client';
import { SchemaDatasetInput, SchemaJob, SchemaTrainJobPayload } from '../../api/openapi-spec';

export type SchemaTrainJob = Omit<SchemaJob, 'payload'> & {
    payload: SchemaTrainJobPayload;
};

interface TrainModelDialogProps {
    title: string;
    submitLabel: string;
    datasets: SchemaDatasetInput[];
    projectId: string;
    defaultName?: string;
    defaultDatasetId?: Key | null;
    defaultMaxSteps?: number;
    extraPayload?: Partial<SchemaTrainJobPayload>;
    policyField?: ReactNode;
    getPolicy: () => string | undefined;
    close: (job: SchemaTrainJob | undefined) => void;
}

export const TrainModelDialog = ({
    title,
    submitLabel,
    datasets,
    projectId,
    defaultName = '',
    defaultDatasetId = null,
    defaultMaxSteps = 100,
    extraPayload,
    policyField,
    getPolicy,
    close,
}: TrainModelDialogProps) => {
    const [name, setName] = useState<string>(defaultName);
    const [selectedDataset, setSelectedDataset] = useState<Key | null>(defaultDatasetId);
    const [maxSteps, setMaxSteps] = useState<number>(defaultMaxSteps);
    const [batchSize, setBatchSize] = useState<number>(8);

    const trainMutation = $api.useMutation('post', '/api/jobs:train');

    const save = () => {
        const dataset_id = selectedDataset?.toString();
        const policy = getPolicy();

        if (!dataset_id || !policy) {
            return;
        }

        const payload: SchemaTrainJobPayload = {
            dataset_id,
            project_id: projectId,
            model_name: name,
            policy,
            max_steps: maxSteps,
            batch_size: batchSize,
            ...extraPayload,
        };
        trainMutation.mutateAsync({ body: payload }).then((response) => {
            close(response as SchemaTrainJob | undefined);
        });
    };

    return (
        <Dialog>
            <Heading>{title}</Heading>
            <Divider />
            <Content>
                <Form
                    onSubmit={(e) => {
                        e.preventDefault();
                        save();
                    }}
                    validationBehavior='native'
                >
                    <TextField label='Name' value={name} onChange={setName} />
                    <Picker label='Dataset' selectedKey={selectedDataset} onSelectionChange={setSelectedDataset}>
                        {datasets.map((dataset) => (
                            <Item key={dataset.id}>{dataset.name}</Item>
                        ))}
                    </Picker>
                    {policyField}
                    <Disclosure isQuiet UNSAFE_style={{ padding: 0 }}>
                        <DisclosureTitle UNSAFE_style={{ fontSize: 13, padding: '4px 0' }}>
                            Advanced settings
                        </DisclosureTitle>
                        <DisclosurePanel UNSAFE_style={{ padding: 0 }}>
                            <Flex direction='row' gap='size-150' width='100%'>
                                <NumberField
                                    label='Max Steps'
                                    value={maxSteps}
                                    onChange={setMaxSteps}
                                    minValue={100}
                                    maxValue={100000}
                                    step={100}
                                    flex
                                />
                                <NumberField
                                    label='Batch Size'
                                    value={batchSize}
                                    onChange={setBatchSize}
                                    minValue={1}
                                    maxValue={256}
                                    step={1}
                                    flex
                                />
                            </Flex>
                        </DisclosurePanel>
                    </Disclosure>
                </Form>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={save}>
                    {submitLabel}
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
