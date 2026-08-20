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

const STREAMING_PRESETS: StreamingPreset[] = [
    {
        id: 'cpu-h264',
        name: 'CPU · H.264',
        description: 'Software encoding with the broadest player compatibility.',
        hardware: 'Multi-core CPUs',
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
        name: 'CPU · H.265',
        description: 'Software encoding with better compression and smaller files.',
        hardware: 'Multi-core CPUs',
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
        name: 'Intel GPU · QSV',
        description: 'Hardware encoding that keeps the CPU mostly free.',
        hardware: 'Intel Arc / Iris Xe GPUs',
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
        name: 'NVIDIA GPU · NVENC',
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

/** Convert a preset string to a number when it is an integer literal, else keep the string. */
const parsePreset = (value: string): number | string | null => {
    const trimmed = value.trim();
    if (trimmed === '') return null;
    const numeric = Number(trimmed);
    return Number.isInteger(numeric) && String(numeric) === trimmed ? numeric : trimmed;
};

const toCurrentValues = (streaming: SchemaStreamingSettings): StreamingPresetValues => ({
    vcodec: streaming.vcodec,
    pix_fmt: streaming.pix_fmt ?? null,
    crf: streaming.crf ?? null,
    preset: streaming.preset ?? null,
    encoder_threads: streaming.encoder_threads ?? null,
    encoder_queue_maxsize: streaming.encoder_queue_maxsize,
});

type ConfigurationOption = {
    id: string;
    name: string;
    description: string;
};

const CONFIGURATION_OPTIONS: ConfigurationOption[] = [
    {
        id: 'custom',
        name: 'Custom',
        description: 'Manually configure each encoder option.',
    },
    ...STREAMING_PRESETS.map((preset) => ({
        id: preset.id,
        name: preset.name,
        description: `${preset.description} Recommended for ${preset.hardware}.`,
    })),
];

/** Find the preset matching the loaded values, or 'custom' when the user has their own config. */
const detectMode = (streaming: SchemaStreamingSettings): string => {
    const current = JSON.stringify(toCurrentValues(streaming));
    return STREAMING_PRESETS.find((preset) => JSON.stringify(preset.values) === current)?.id ?? 'custom';
};

/** Recommend a recording preset for the hardware the Studio host reports. */
const recommendPreset = (devices: SchemaDeviceInfo[] | undefined): string | undefined => {
    if (devices === undefined || devices.length === 0) {
        return undefined;
    }
    if (devices.some((device) => device.type === 'cuda')) {
        return 'nvidia-nvenc';
    }
    if (devices.some((device) => device.type === 'xpu')) {
        return 'intel-qsv';
    }
    return 'cpu-h264';
};

type StreamingSettingsFormProps = {
    streaming: SchemaStreamingSettings;
};

export const StreamingSettingsForm = ({ streaming }: StreamingSettingsFormProps) => {
    const patchMutation = useSettingsPatch();

    const [mode, setMode] = useState(() => detectMode(streaming));
    const [vcodec, setVcodec] = useState(streaming.vcodec);
    const [pixFmt, setPixFmt] = useState(streaming.pix_fmt ?? '');
    const [crf, setCrf] = useState<number | null>(streaming.crf ?? null);
    const [preset, setPreset] = useState(streaming.preset === null ? '' : String(streaming.preset));
    const [encoderThreads, setEncoderThreads] = useState<number | null>(streaming.encoder_threads ?? null);
    const [encoderQueueMaxsize, setEncoderQueueMaxsize] = useState(streaming.encoder_queue_maxsize);
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);

    const readOnly = mode !== 'custom';
    const selectedPreset = STREAMING_PRESETS.find((candidate) => candidate.id === mode);

    const { data: trainingDevices } = $api.useQuery('get', '/api/system/devices/training');
    const recommendedPresetId = recommendPreset(trainingDevices?.devices);
    const recommendedPreset = STREAMING_PRESETS.find((candidate) => candidate.id === recommendedPresetId);
    const showRecommendation = recommendedPreset !== undefined && recommendedPreset.id !== mode;

    const markDirty = () => {
        setDirty(true);
        setSaved(false);
    };

    const selectMode = (key: string) => {
        setMode(key);
        markDirty();
        const selected = STREAMING_PRESETS.find((candidate) => candidate.id === key);
        if (selected === undefined) {
            return;
        }
        setVcodec(selected.values.vcodec);
        setPixFmt(selected.values.pix_fmt ?? '');
        setCrf(selected.values.crf);
        setPreset(selected.values.preset === null ? '' : String(selected.values.preset));
        setEncoderThreads(selected.values.encoder_threads);
        setEncoderQueueMaxsize(selected.values.encoder_queue_maxsize);
    };

    const save = () => {
        const body: SchemaSettingsUpdate = {
            streaming: {
                vcodec,
                pix_fmt: pixFmt === '' ? null : pixFmt,
                crf,
                preset: parsePreset(preset),
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

    return (
        <SettingsSection
            title='Streaming'
            description='Video encoding for dataset recordings.'
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
                        <Text slot='description'>{option.description}</Text>
                    </Item>
                )}
            </Picker>
            {showRecommendation && recommendedPreset !== undefined && (
                <Flex UNSAFE_className={classes.recommendation} alignItems='center' gap='size-150' wrap>
                    <Text UNSAFE_className={classes.presetDescription}>
                        Based on the detected hardware, we recommend {recommendedPreset.name} for{' '}
                        {recommendedPreset.hardware}.
                    </Text>
                    <ActionButton onPress={() => selectMode(recommendedPreset.id)}>Apply</ActionButton>
                </Flex>
            )}
            {readOnly && selectedPreset !== undefined && (
                <Text UNSAFE_className={classes.presetDescription}>
                    {selectedPreset.description} Recommended for {selectedPreset.hardware}. Fields are read-only while a
                    preset is selected.
                </Text>
            )}
            {!readOnly && <Text UNSAFE_className={classes.presetDescription}>Fields are editable in Custom mode.</Text>}
            <TextField
                label='Video codec'
                value={vcodec}
                isReadOnly={readOnly}
                onChange={(value) => {
                    setVcodec(value);
                    markDirty();
                }}
                width='100%'
            />
            <TextField
                label='Pixel format'
                value={pixFmt}
                isReadOnly={readOnly}
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
                isReadOnly={readOnly}
                onChange={(value) => {
                    setCrf(Number.isNaN(value) ? null : value);
                    markDirty();
                }}
                width='100%'
            />
            <TextField
                label='Preset'
                value={preset}
                isReadOnly={readOnly}
                onChange={(value) => {
                    setPreset(value);
                    markDirty();
                }}
                placeholder='e.g. veryfast or a number'
                width='100%'
            />
            <NumberField
                label='Encoder threads'
                value={encoderThreads ?? undefined}
                isReadOnly={readOnly}
                onChange={(value) => {
                    setEncoderThreads(Number.isNaN(value) ? null : value);
                    markDirty();
                }}
                width='100%'
            />
            <NumberField
                label='Encoder queue size'
                value={encoderQueueMaxsize}
                isReadOnly={readOnly}
                onChange={(value) => {
                    if (!Number.isNaN(value)) {
                        setEncoderQueueMaxsize(value);
                        markDirty();
                    }
                }}
                width='100%'
            />
        </SettingsSection>
    );
};
