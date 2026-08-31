import { useState } from 'react';

import { ActionButton, Flex, Item, NumberField, Picker, Text, TextField } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaDeviceInfo, SchemaSettingsUpdate, SchemaStreamingSettings } from '../../../api/openapi-spec';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

import classes from './general-settings.module.css';

type StreamingPresetValues = {
    vcodec: string;
    pix_fmt: string | null;
    crf: number | null;
    preset: number | string | null;
    encoder_threads: number | null;
    encoder_queue_maxsize: number;
};

type StreamingPreset = {
    id: string;
    name: string;
    description: string;
    hardware: string;
    values: StreamingPresetValues;
};

type ConfigurationOption = { id: string; name: string; description: string };

const STREAMING_PRESETS: StreamingPreset[] = [
    {
        id: 'cpu-h264',
        name: 'CPU H.264',
        description: 'Software encoding with the broadest player compatibility.',
        hardware: 'multi-core CPUs',
        values: {
            vcodec: 'libx264',
            pix_fmt: null,
            crf: 23,
            preset: 'veryfast',
            encoder_threads: null,
            encoder_queue_maxsize: 60,
        },
    },
    {
        id: 'cpu-h265',
        name: 'CPU H.265',
        description: 'Software encoding with better compression and smaller files.',
        hardware: 'multi-core CPUs',
        values: {
            vcodec: 'libx265',
            pix_fmt: null,
            crf: 28,
            preset: 'veryfast',
            encoder_threads: null,
            encoder_queue_maxsize: 60,
        },
    },
    {
        id: 'intel-qsv',
        name: 'Intel GPU QSV',
        description: 'Hardware encoding that keeps the CPU mostly free.',
        hardware: 'Intel Arc or Iris Xe GPUs',
        values: {
            vcodec: 'h264_qsv',
            pix_fmt: 'nv12',
            crf: 23,
            preset: 'veryfast',
            encoder_threads: null,
            encoder_queue_maxsize: 60,
        },
    },
    {
        id: 'nvidia-nvenc',
        name: 'NVIDIA GPU NVENC',
        description: 'Hardware encoding that keeps the CPU mostly free.',
        hardware: 'NVIDIA GeForce RTX GPUs',
        values: {
            vcodec: 'h264_nvenc',
            pix_fmt: 'nv12',
            crf: 23,
            preset: 'p5',
            encoder_threads: null,
            encoder_queue_maxsize: 60,
        },
    },
];

const CONFIGURATION_OPTIONS: ConfigurationOption[] = [
    { id: 'custom', name: 'Custom', description: 'Manually configure each encoder option.' },
    ...STREAMING_PRESETS.map((preset) => ({
        id: preset.id,
        name: preset.name,
        description:
            preset.id === 'intel-qsv' || preset.id === 'nvidia-nvenc'
                ? `${preset.description} Uses hardware acceleration on ${preset.hardware}.`
                : `${preset.description} Uses GPL-licensed software encoding on ${preset.hardware}.`,
    })),
];

const presetValue = (value: string): number | string | null => {
    const trimmed = value.trim();
    if (trimmed === '') return null;
    const numeric = Number(trimmed);
    return Number.isInteger(numeric) && String(numeric) === trimmed ? numeric : trimmed;
};

const currentValues = (streaming: SchemaStreamingSettings): StreamingPresetValues => ({
    vcodec: streaming.vcodec,
    pix_fmt: streaming.pix_fmt ?? null,
    crf: streaming.crf ?? null,
    preset: streaming.preset ?? null,
    encoder_threads: streaming.encoder_threads ?? null,
    encoder_queue_maxsize: streaming.encoder_queue_maxsize,
});

const matchingPreset = (streaming: SchemaStreamingSettings): string => {
    const values = JSON.stringify(currentValues(streaming));
    return STREAMING_PRESETS.find((preset) => JSON.stringify(preset.values) === values)?.id ?? 'custom';
};

const recommendedPreset = (devices: SchemaDeviceInfo[] | undefined): string | undefined => {
    if (devices?.some((device) => device.type === 'cuda')) return 'nvidia-nvenc';
    if (devices?.some((device) => device.type === 'xpu')) return 'intel-qsv';
    return devices?.length ? 'cpu-h264' : undefined;
};

type StreamingSettingsFormProps = { streaming: SchemaStreamingSettings };

export const StreamingSettingsForm = ({ streaming }: StreamingSettingsFormProps) => {
    const patchMutation = useSettingsPatch();
    const [mode, setMode] = useState(() => matchingPreset(streaming));
    const [vcodec, setVcodec] = useState(streaming.vcodec);
    const [pixFmt, setPixFmt] = useState(streaming.pix_fmt ?? '');
    const [crf, setCrf] = useState<number | null>(streaming.crf ?? null);
    const [encoderPreset, setEncoderPreset] = useState(streaming.preset === null ? '' : String(streaming.preset));
    const [encoderThreads, setEncoderThreads] = useState<number | null>(streaming.encoder_threads ?? null);
    const [encoderQueueMaxsize, setEncoderQueueMaxsize] = useState(streaming.encoder_queue_maxsize);
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);
    const { data: trainingDevices } = $api.useQuery('get', '/api/system/devices/training');
    const recommendedMode = recommendedPreset(trainingDevices?.devices);
    const selectedPreset = STREAMING_PRESETS.find((preset) => preset.id === mode);
    const isCustom = mode === 'custom';

    const markDirty = () => {
        setDirty(true);
        setSaved(false);
    };

    const save = () => {
        const body: SchemaSettingsUpdate = {
            streaming: {
                vcodec,
                pix_fmt: pixFmt === '' ? null : pixFmt,
                crf,
                preset: presetValue(encoderPreset),
                encoder_threads: encoderThreads,
                encoder_queue_maxsize: encoderQueueMaxsize,
            },
        };
        patchMutation.mutate(
            { body },
            {
                onSuccess: () => {
                    setDirty(false);
                    setSaved(true);
                },
            }
        );
    };

    const selectMode = (key: string) => {
        setMode(key);
        markDirty();
        const selected = STREAMING_PRESETS.find((preset) => preset.id === key);
        if (selected === undefined) return;
        setVcodec(selected.values.vcodec);
        setPixFmt(selected.values.pix_fmt ?? '');
        setCrf(selected.values.crf);
        setEncoderPreset(selected.values.preset === null ? '' : String(selected.values.preset));
        setEncoderThreads(selected.values.encoder_threads);
        setEncoderQueueMaxsize(selected.values.encoder_queue_maxsize);
    };

    const customize = () => {
        setMode('custom');
    };

    const summaryRows = [
        { label: 'Video codec', value: vcodec },
        { label: 'Pixel format', value: pixFmt === '' ? 'Encoder default' : pixFmt },
        { label: 'CRF', value: crf === null ? 'Encoder default' : String(crf) },
        { label: 'Preset', value: encoderPreset === '' ? 'Encoder default' : encoderPreset },
        { label: 'Encoder threads', value: encoderThreads === null ? 'Auto' : String(encoderThreads) },
        { label: 'Encoder queue size', value: String(encoderQueueMaxsize) },
    ];

    return (
        <SettingsSection
            title='Streaming'
            description='Video encoding defaults for new dataset recordings.'
            isDirty={dirty}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            <Picker
                label='Configuration'
                width='100%'
                selectedKey={mode}
                items={CONFIGURATION_OPTIONS}
                onSelectionChange={(key) => selectMode(String(key))}
            >
                {(option) => (
                    <Item key={option.id} textValue={option.name}>
                        <Text>{option.name}</Text>
                        <Text slot='description'>
                            {option.id === recommendedMode ? `Recommended · ${option.description}` : option.description}
                        </Text>
                    </Item>
                )}
            </Picker>
            {!isCustom && selectedPreset !== undefined ? (
                <>
                    <Text UNSAFE_className={classes.presetDescription}>{selectedPreset.description}</Text>
                    <Flex direction='column' gap='size-100' UNSAFE_className={classes.summary}>
                        {summaryRows.map((row) => (
                            <Flex key={row.label} justifyContent='space-between' gap='size-200'>
                                <Text UNSAFE_className={classes.summaryLabel}>{row.label}</Text>
                                <Text UNSAFE_className={classes.summaryValue}>{row.value}</Text>
                            </Flex>
                        ))}
                    </Flex>
                    <Flex alignItems='center' gap='size-200'>
                        <ActionButton onPress={customize}>Customize</ActionButton>
                        <Text UNSAFE_className={classes.customizeHint}>Adjust the preset values in custom mode.</Text>
                    </Flex>
                </>
            ) : (
                <>
                    <TextField
                        label='Video codec'
                        value={vcodec}
                        onChange={(value) => {
                            setVcodec(value);
                            markDirty();
                        }}
                        width='100%'
                    />
                    <TextField
                        label='Pixel format'
                        value={pixFmt}
                        onChange={(value) => {
                            setPixFmt(value);
                            markDirty();
                        }}
                        placeholder='Leave empty to let the encoder pick'
                        width='100%'
                    />
                    <NumberField
                        label='CRF'
                        value={crf ?? undefined}
                        onChange={(value) => {
                            setCrf(Number.isNaN(value) ? null : value);
                            markDirty();
                        }}
                        width='100%'
                    />
                    <TextField
                        label='Preset'
                        value={encoderPreset}
                        onChange={(value) => {
                            setEncoderPreset(value);
                            markDirty();
                        }}
                        placeholder='e.g. veryfast or a number'
                        width='100%'
                    />
                    <NumberField
                        label='Encoder threads'
                        value={encoderThreads ?? undefined}
                        onChange={(value) => {
                            setEncoderThreads(Number.isNaN(value) ? null : value);
                            markDirty();
                        }}
                        width='100%'
                    />
                    <NumberField
                        label='Encoder queue size'
                        value={encoderQueueMaxsize}
                        onChange={(value) => {
                            if (!Number.isNaN(value)) {
                                setEncoderQueueMaxsize(value);
                                markDirty();
                            }
                        }}
                        width='100%'
                    />
                </>
            )}
        </SettingsSection>
    );
};
