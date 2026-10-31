import { SchemaPolicyTrainingConfiguration } from '../../api/openapi-spec';

export const getMockedPolicyTrainingConfiguration = (
    overrides: Partial<SchemaPolicyTrainingConfiguration> = {}
): SchemaPolicyTrainingConfiguration => ({
    defaults: {
        batch_size: 8,
        max_epochs: 5,
        max_steps: null,
        num_workers: 'auto',
        precision: '32-true',
        compile_model: false,
        gradient_checkpointing: null,
        auto_scale_batch_size: false,
        augment_images: false,
        validation: { split: 0.1, seed: null },
        lora: null,
        snapflow: null,
    },
    help: {},
    camera_slots: [],
    ...overrides,
});
