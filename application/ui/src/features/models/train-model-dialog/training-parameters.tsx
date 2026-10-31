import {
    Checkbox,
    Content,
    ContextualHelp,
    Flex,
    Heading,
    Item,
    Key,
    NumberField,
    Picker,
    Slider,
    Switch,
    Text,
} from '@geti-ui/ui';

import { SchemaPolicyTrainingHelp } from '../../../api/openapi-spec';

export const MIN_BATCH_SIZE_EXPONENT = 0;
export const MAX_BATCH_SIZE_EXPONENT = 8;

// The slider moves in exponents so every stop is a power of two (1, 2, 4, ... 256),
// which is what the trainer expects; the label maps back to the batch size itself.
export const batchSizeToExponent = (batchSize: number): number => {
    const exponent = Math.round(Math.log2(Math.max(batchSize, 1)));

    return Math.min(Math.max(exponent, MIN_BATCH_SIZE_EXPONENT), MAX_BATCH_SIZE_EXPONENT);
};

export const exponentToBatchSize = (exponent: number): number => 2 ** exponent;

/**
 * Distillation needs at least one epoch to train the policy it distils, plus
 * one to distil it, so a single-epoch run cannot use SnapFlow at all.
 */
export const MIN_EPOCHS_FOR_SNAPFLOW = 2;

export const PRECISION_LABELS: Record<string, string> = {
    'bf16-mixed': 'BF16 Mixed',
    'bf16-true': 'BF16 True',
    '32-true': '32-bit',
};

interface TrainingParametersProps {
    help?: SchemaPolicyTrainingHelp;
    maxEpochs: number;
    onMaxEpochsChange: (value: number) => void;
    batchSize: number;
    trainingSteps?: number;
    onBatchSizeChange: (value: number) => void;
    numWorkers: Key | null;
    onNumWorkersChange: (value: Key | null) => void;
    autoScaleBatchSize: boolean;
    onAutoScaleBatchSizeChange: (value: boolean) => void;
    precision: Key | null;
    onPrecisionChange: (value: Key | null) => void;
    compileModel: boolean;
    onCompileModelChange: (value: boolean) => void;
    gradientCheckpointing: boolean | null;
    onGradientCheckpointingChange: (value: boolean) => void;
    augmentImages: boolean;
    onAugmentImagesChange: (value: boolean) => void;
    isAutoScaleBatchDisabled: boolean;
    recommendedPrecision?: string;
    /** False for policies without a LoRA/DoRA mixin; hides the LoRA controls entirely. */
    isLoraSupported: boolean;
    loraEnabled: boolean;
    onLoraEnabledChange: (value: boolean) => void;
    loraRank: number;
    onLoraRankChange: (value: number) => void;
    loraAlpha: number | null;
    onLoraAlphaChange: (value: number | null) => void;
    loraDropout: number;
    onLoraDropoutChange: (value: number) => void;
    loraUseDora: boolean;
    onLoraUseDoraChange: (value: boolean) => void;
    /** False for policies without a flow-matching sampler; hides the SnapFlow controls entirely. */
    isSnapflowSupported: boolean;
    snapflowEnabled: boolean;
    onSnapflowEnabledChange: (value: boolean) => void;
    snapflowDistillEpochs: number;
    onSnapflowDistillEpochsChange: (value: number) => void;
}

export const TrainingParameters = ({
    help = {},
    maxEpochs,
    onMaxEpochsChange,
    batchSize,
    trainingSteps,
    onBatchSizeChange,
    numWorkers,
    onNumWorkersChange,
    autoScaleBatchSize,
    onAutoScaleBatchSizeChange,
    precision,
    onPrecisionChange,
    compileModel,
    onCompileModelChange,
    gradientCheckpointing,
    onGradientCheckpointingChange,
    augmentImages,
    onAugmentImagesChange,
    isAutoScaleBatchDisabled,
    recommendedPrecision,
    isLoraSupported,
    loraEnabled,
    onLoraEnabledChange,
    loraRank,
    onLoraRankChange,
    loraAlpha,
    onLoraAlphaChange,
    loraDropout,
    onLoraDropoutChange,
    loraUseDora,
    onLoraUseDoraChange,
    isSnapflowSupported,
    snapflowEnabled,
    onSnapflowEnabledChange,
    snapflowDistillEpochs,
    onSnapflowDistillEpochsChange,
}: TrainingParametersProps) => (
    <Flex direction='column' gap='size-150' width='100%'>
        <Flex direction='row' gap='size-150' width='100%'>
            <Flex direction='row' gap='size-300' width='100%'>
                <Slider
                    label='Batch Size'
                    value={batchSizeToExponent(batchSize)}
                    onChange={(exponent) => onBatchSizeChange(exponentToBatchSize(exponent))}
                    getValueLabel={(exponent) => `${exponentToBatchSize(exponent)}`}
                    minValue={MIN_BATCH_SIZE_EXPONENT}
                    maxValue={MAX_BATCH_SIZE_EXPONENT}
                    step={1}
                    flex
                    isFilled
                    isDisabled={autoScaleBatchSize}
                    contextualHelp={
                        <ContextualHelp variant='info' aria-label='Batch size help'>
                            <Heading>Batch size</Heading>
                            <Content>
                                <Text>{help.batch_size ?? 'Samples per step. Larger batches need more memory.'}</Text>
                            </Content>
                        </ContextualHelp>
                    }
                />
                <Flex direction='row' gap='size-100' alignItems='center'>
                    <Switch
                        isEmphasized
                        isSelected={autoScaleBatchSize}
                        onChange={onAutoScaleBatchSizeChange}
                        isDisabled={isAutoScaleBatchDisabled}
                    >
                        Auto
                    </Switch>
                    <ContextualHelp variant='info'>
                        <Heading>Auto scale batch size</Heading>
                        <Content>
                            <Text>
                                {help.auto_scale_batch_size ??
                                    'Finds a fitting batch size on CUDA. Ignores the slider.'}
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
            </Flex>
        </Flex>
        <Flex direction='row' gap='size-150' width='100%'>
            <NumberField
                label='Max Epochs'
                description={
                    trainingSteps === undefined
                        ? undefined
                        : autoScaleBatchSize
                          ? 'Training steps: set when training starts (Auto batch size)'
                          : `Estimated training steps: ${trainingSteps.toLocaleString()}`
                }
                value={maxEpochs}
                onChange={onMaxEpochsChange}
                minValue={1}
                maxValue={10_000}
                step={1}
                width='100%'
                contextualHelp={
                    <ContextualHelp variant='info'>
                        <Heading>Max epochs</Heading>
                        <Content>
                            <Text>{help.max_epochs ?? "Full dataset passes. Start with this policy's defaults."}</Text>
                        </Content>
                    </ContextualHelp>
                }
            />
            <Picker
                width='100%'
                label='Data Workers'
                selectedKey={numWorkers}
                onSelectionChange={onNumWorkersChange}
                contextualHelp={
                    <ContextualHelp variant='info'>
                        <Heading>Data workers</Heading>
                        <Content>
                            <Text>{help.num_workers ?? 'Parallel data loaders. Auto uses available CPU cores.'}</Text>
                        </Content>
                    </ContextualHelp>
                }
            >
                <Item key='auto'>Auto</Item>
                <Item key='0'>0 (main process)</Item>
                <Item key='1'>1</Item>
                <Item key='2'>2</Item>
                <Item key='4'>4</Item>
                <Item key='8'>8</Item>
                <Item key='16'>16</Item>
            </Picker>
        </Flex>
        <Flex direction='row' gap='size-200' width='100%'>
            <Picker
                label='Precision'
                description={
                    recommendedPrecision
                        ? `${PRECISION_LABELS[recommendedPrecision]} recommended by this policy's defaults`
                        : undefined
                }
                selectedKey={precision}
                onSelectionChange={onPrecisionChange}
                contextualHelp={
                    <ContextualHelp variant='info'>
                        <Heading>Training precision</Heading>
                        <Content>
                            <Text>
                                {help.precision ?? (
                                    <>
                                        BF16 Mixed uses half-precision where safe to reduce memory use. BF16 True runs
                                        entirely in BF16. FP32 uses full precision for numerical stability.
                                    </>
                                )}
                            </Text>
                        </Content>
                    </ContextualHelp>
                }
            >
                <Item key='bf16-mixed'>BF16 Mixed</Item>
                <Item key='bf16-true'>BF16 True</Item>
                <Item key='32-true'>32-bit</Item>
            </Picker>
        </Flex>
        <Flex direction='row' gap='size-200' width='100%'>
            <Checkbox isEmphasized isSelected={compileModel} onChange={onCompileModelChange}>
                Compile model
            </Checkbox>
            <ContextualHelp variant='info'>
                <Heading>Compile model</Heading>
                <Content>
                    <Text>
                        {help.compile_model ?? 'Compiles the model for faster training, with extra startup time.'}
                    </Text>
                </Content>
            </ContextualHelp>
        </Flex>
        {gradientCheckpointing !== null && (
            <Flex direction='row' gap='size-200' alignItems='center'>
                <Checkbox isEmphasized isSelected={gradientCheckpointing} onChange={onGradientCheckpointingChange}>
                    Gradient checkpointing
                </Checkbox>
                <ContextualHelp variant='info'>
                    <Heading>Gradient checkpointing</Heading>
                    <Content>
                        <Text>
                            {help.gradient_checkpointing ??
                                'Recomputes activations to save memory, using extra compute.'}
                        </Text>
                    </Content>
                </ContextualHelp>
            </Flex>
        )}
        <Flex direction='row' gap='size-200' width='100%'>
            <Flex direction='row' alignItems='center'>
                <Checkbox isEmphasized isSelected={augmentImages} onChange={onAugmentImagesChange}>
                    Augment images
                </Checkbox>
                <ContextualHelp variant='info'>
                    <Heading>Augment images</Heading>
                    <Content>
                        <Text>
                            {help.augment_images ?? (
                                <>
                                    Varies lighting, sharpness and small rotations to reduce dependence on the recording
                                    setup. This does not always improve results and makes training slightly slower.
                                    Validation images are left unchanged.
                                </>
                            )}
                        </Text>
                    </Content>
                </ContextualHelp>
            </Flex>
        </Flex>
        {isLoraSupported && (
            <Flex direction='column' gap='size-150' width='100%'>
                <Flex direction='row' gap='size-100' alignItems='center'>
                    <Checkbox isEmphasized isSelected={loraEnabled} onChange={onLoraEnabledChange}>
                        LoRA fine-tuning
                    </Checkbox>
                    <ContextualHelp variant='info'>
                        <Heading>LoRA fine-tuning</Heading>
                        <Content>
                            <Text>
                                {help.lora?.description ?? (
                                    <>
                                        Freezes base weights and trains low-rank adapters to save memory, at the cost of
                                        some capacity versus full fine-tuning. The library automatically scales the
                                        learning rate for adapter training.
                                    </>
                                )}
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
                {loraEnabled && (
                    <Flex direction='row' gap='size-150' width='100%'>
                        <NumberField
                            label='LoRA rank'
                            value={loraRank}
                            onChange={onLoraRankChange}
                            minValue={8}
                            maxValue={256}
                            step={8}
                            width='100%'
                            contextualHelp={
                                <ContextualHelp variant='info'>
                                    <Heading>LoRA rank</Heading>
                                    <Content>
                                        <Text>
                                            {help.lora?.rank ?? (
                                                <>
                                                    Higher rank increases adapter capacity and memory use. Start at 32;
                                                    16 is lighter, and 64 gives more capacity for larger datasets.
                                                </>
                                            )}
                                        </Text>
                                    </Content>
                                </ContextualHelp>
                            }
                        />
                        <NumberField
                            label='LoRA alpha'
                            value={loraAlpha ?? undefined}
                            onChange={onLoraAlphaChange}
                            minValue={1}
                            width='100%'
                            contextualHelp={
                                <ContextualHelp variant='info'>
                                    <Heading>LoRA alpha</Heading>
                                    <Content>
                                        <Text>
                                            {help.lora?.alpha ?? 'Scaling = alpha / rank. Empty uses the rank.'}
                                        </Text>
                                    </Content>
                                </ContextualHelp>
                            }
                        />
                        <NumberField
                            label='LoRA dropout'
                            value={loraDropout}
                            onChange={onLoraDropoutChange}
                            minValue={0}
                            maxValue={0.99}
                            step={0.01}
                            width='100%'
                            contextualHelp={
                                <ContextualHelp variant='info'>
                                    <Heading>LoRA dropout</Heading>
                                    <Content>
                                        <Text>
                                            {help.lora?.dropout ??
                                                'Dropout probability applied to LoRA adapter inputs.'}
                                        </Text>
                                    </Content>
                                </ContextualHelp>
                            }
                        />
                        <Flex direction='column' gap='size-150' width='100%' justifyContent='center'>
                            <Flex direction='row' gap='size-100' alignItems='center'>
                                <Checkbox isEmphasized isSelected={loraUseDora} onChange={onLoraUseDoraChange}>
                                    Use DoRA
                                </Checkbox>
                                <ContextualHelp variant='info'>
                                    <Heading>Use DoRA</Heading>
                                    <Content>
                                        <Text>
                                            {help.lora?.use_dora ??
                                                'Weight-decomposed LoRA; uses extra compute and memory.'}
                                        </Text>
                                    </Content>
                                </ContextualHelp>
                            </Flex>
                        </Flex>
                    </Flex>
                )}
            </Flex>
        )}
        {isSnapflowSupported && (
            <Flex direction='row' gap='size-150' width='100%' alignItems='end'>
                <Flex direction='column' gap='size-150' width='100%' justifyContent='center'>
                    <Flex direction='row' gap='size-100' alignItems='center'>
                        {/* A single-epoch run has no epoch left over to train the policy that gets distilled. */}
                        <Checkbox
                            isEmphasized
                            isSelected={snapflowEnabled}
                            onChange={onSnapflowEnabledChange}
                            isDisabled={maxEpochs < MIN_EPOCHS_FOR_SNAPFLOW}
                        >
                            SnapFlow distillation
                        </Checkbox>
                        <ContextualHelp variant='info'>
                            <Heading>SnapFlow distillation</Heading>
                            <Content>
                                <Text>
                                    {help.snapflow?.description ??
                                        'Adds distillation after training for one-step inference.'}
                                </Text>
                            </Content>
                        </ContextualHelp>
                    </Flex>
                </Flex>
                <NumberField
                    label='Distillation epochs'
                    value={snapflowDistillEpochs}
                    onChange={onSnapflowDistillEpochsChange}
                    minValue={1}
                    maxValue={10_000}
                    step={1}
                    width='100%'
                    isDisabled={!snapflowEnabled || maxEpochs < MIN_EPOCHS_FOR_SNAPFLOW}
                    contextualHelp={
                        <ContextualHelp variant='info'>
                            <Heading>Distillation epochs</Heading>
                            <Content>
                                <Text>
                                    {help.snapflow?.distill_epochs ??
                                        'Extra epochs after normal training. Start with 2-5.'}
                                </Text>
                            </Content>
                        </ContextualHelp>
                    }
                />
            </Flex>
        )}
    </Flex>
);
