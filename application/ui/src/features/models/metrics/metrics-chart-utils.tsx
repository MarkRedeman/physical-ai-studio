import { ChartTooltipContentProps, LineChartSeriesConfig, useChartsTheme } from '@geti-ui/charts';
import { Flex, View } from '@geti-ui/ui';
import type { CSSProperties, ReactNode } from 'react';

import { Box } from '../../../routes/models/box.component';

export interface MetricsEntry {
    epoch: number | null;
    step: number | null;
    system_accelerator_memory_mb: number | null;
    system_accelerator_memory_percent?: number | null;
    system_accelerator_memory_total_mb?: number | null;
    system_accelerator_power_w?: number | null;
    system_accelerator_utilization_percent?: number | null;
    system_step_time_s: number | null;
    train_action_error_epoch?: number | null;
    train_action_error_step?: number | null;
    train_epoch: number | null;
    train_fractional_epoch: number | null;
    train_loss: number | null;
    train_loss_epoch: number | null;
    train_loss_step: number | null;
    train_lr: number | null;
    train_grad_norm: number | null;
    val_action_error?: number | null;
    val_epoch: number | null;
    val_loss: number | null;
}

export type NumericMetricKey =
    | 'epoch'
    | 'step'
    | 'system_accelerator_memory_mb'
    | 'system_accelerator_memory_percent'
    | 'system_accelerator_memory_total_mb'
    | 'system_accelerator_power_w'
    | 'system_accelerator_utilization_percent'
    | 'system_step_time_s'
    | 'train_action_error_epoch'
    | 'train_action_error_step'
    | 'train_epoch'
    | 'train_fractional_epoch'
    | 'train_loss'
    | 'train_loss_epoch'
    | 'train_loss_step'
    | 'train_lr'
    | 'train_grad_norm'
    | 'val_action_error'
    | 'val_epoch'
    | 'val_loss';

export type MetricChartPoint = {
    x: number;
    y: number;
    epoch?: number;
    step?: number | null;
};

export type MetricChartSeries = {
    dataKey: string;
    name: string;
    data: MetricChartPoint[];
    color?: string;
    dashed?: boolean;
    curve?: LineChartSeriesConfig['curve'];
};

type ChartRow = {
    epoch: number;
    step?: number | null;
    stepsBySeries?: Record<string, number>;
    [key: string]: number | null | Record<string, number> | undefined;
};

type RuntimeTooltipEntry = {
    name?: string;
    value?: number | string;
    color?: string;
    unit?: string;
    dataKey?: string;
    payload?: ChartRow;
};

type MetricTooltipProps = ChartTooltipContentProps & {
    valueFormatter?: (value: number) => string;
};

export const CHART_HEIGHT = 300;
export const CHART_MARGIN = { top: 35, right: 35, bottom: 35, left: 35 };
export const CHART_THEME = { dotRadius: 0, activeDotRadius: 0 };
export const CHART_HIGHLIGHT = { enabled: true, interaction: { legendHover: true, legendClick: true } };
export const X_AXIS_TICK_COUNT = 8;
export const Y_AXIS_TICK_COUNT = 4;
export const TRAIN_COLOR = 'var(--energy-blue)';
export const STEP_X_KEY = 'train_fractional_epoch';
export const TRAIN_EPOCH_X_KEY = 'train_epoch';
export const VAL_EPOCH_X_KEY = 'val_epoch';

const roundTickValue = (value: number) => Number(value.toFixed(6));

const toEpochValues = (data: MetricsEntry[] | undefined) => {
    if (!data) {
        return [];
    }

    return data.flatMap(({ train_fractional_epoch, train_epoch, val_epoch }) => {
        return [train_fractional_epoch, train_epoch, val_epoch].filter((value): value is number => typeof value === 'number');
    });
};

export const getEquidistantEpochTicks = (data: MetricsEntry[] | undefined, tickCount = X_AXIS_TICK_COUNT) => {
    const epochValues = toEpochValues(data);

    if (epochValues.length === 0) {
        return [0];
    }

    const minEpoch = Math.min(...epochValues);
    const maxEpoch = Math.max(...epochValues);

    if (minEpoch === maxEpoch) {
        return [roundTickValue(minEpoch)];
    }

    const safeTickCount = Math.max(2, tickCount);
    const step = (maxEpoch - minEpoch) / (safeTickCount - 1);

    return Array.from({ length: safeTickCount }, (_, index) => roundTickValue(minEpoch + index * step));
};

export const formatEpochTick = (value: number) => {
    const rounded = Number(value.toFixed(3));

    return rounded.toString();
};

export const formatMetricValue = (value: number | string | readonly (number | string)[] | undefined) => {
    if (typeof value !== 'number') {
        return value;
    }

    return value.toPrecision(4);
};

export const formatAxisValue = (value: number) => formatMetricValue(value)?.toString() ?? '';
export const formatScientific = (value: number) => value.toExponential(2);

export const getFormattedValue = (
    value: number | string | readonly (number | string)[] | undefined,
    valueFormatter?: (value: number) => string
) => {
    return typeof value === 'number' && valueFormatter !== undefined ? valueFormatter(value) : formatMetricValue(value);
};

const formatTooltipLabel = (label: unknown, step?: number) => {
    const epoch = typeof label === 'number' ? Math.floor(label) : label;

    if (typeof step === 'number') {
        return `Step ${step} (epoch = ${epoch})`;
    }

    return `Epoch ${epoch}`;
};

export const toPoints = (data: MetricsEntry[] | undefined, xKey: NumericMetricKey, yKey: NumericMetricKey) => {
    if (!data) {
        return [];
    }

    return data.flatMap((entry): MetricChartPoint[] => {
        const x = entry[xKey];
        const y = entry[yKey];

        if (typeof x !== 'number' || typeof y !== 'number') {
            return [];
        }

        return [{ x, y, step: entry.step }];
    });
};

export const smooth = (points: MetricChartPoint[], alpha = 0.9) => {
    let previous: number | undefined;

    return points.map((point) => {
        previous = previous === undefined ? point.y : alpha * previous + (1 - alpha) * point.y;

        return { ...point, y: previous };
    });
};

export const hasData = (series: MetricChartSeries) => series.data.length > 0;

export const hasAnyMetric = (data: MetricsEntry[] | undefined, key: NumericMetricKey) => {
    return data?.some((entry) => typeof entry[key] === 'number') ?? false;
};

export const buildChartData = (series: MetricChartSeries[], fillGaps = true) => {
    const rowsByX = new Map<number, ChartRow>();

    series.forEach((metricSeries) => {
        for (const point of metricSeries.data) {
            const row = rowsByX.get(point.x) ?? { epoch: point.epoch ?? point.x };
            row.epoch = point.epoch ?? row.epoch;
            row[metricSeries.dataKey] = point.y;
            if (typeof point.step === 'number') {
                row.step = point.step;
                row.stepsBySeries = { ...row.stepsBySeries, [metricSeries.dataKey]: point.step };
            }
            rowsByX.set(point.x, row);
        }
    });

    const rows = Array.from(rowsByX.values()).sort((a, b) => a.epoch - b.epoch);

    if (fillGaps) {
        series.forEach((metricSeries) => fillSeriesGaps(rows, metricSeries.dataKey));
    }

    return rows;
};

const fillSeriesGaps = (rows: ChartRow[], dataKey: string) => {
    const knownIndexes = rows.flatMap((row, index) => (typeof row[dataKey] === 'number' ? [index] : []));

    for (let i = 0; i < knownIndexes.length - 1; i++) {
        const startIndex = knownIndexes[i];
        const endIndex = knownIndexes[i + 1];
        const start = rows[startIndex];
        const end = rows[endIndex];
        const startValue = start[dataKey];
        const endValue = end[dataKey];

        if (typeof startValue !== 'number' || typeof endValue !== 'number' || start.epoch === end.epoch) {
            continue;
        }

        for (let rowIndex = startIndex + 1; rowIndex < endIndex; rowIndex++) {
            const row = rows[rowIndex];
            const ratio = (row.epoch - start.epoch) / (end.epoch - start.epoch);
            row[dataKey] = startValue + ratio * (endValue - startValue);
        }
    }
};

export const buildChartSeries = (series: MetricChartSeries[]) => {
    return series.map((metricSeries): LineChartSeriesConfig => {
        return {
            dataKey: metricSeries.dataKey,
            name: metricSeries.name,
            color: metricSeries.color,
            dashed: metricSeries.dashed,
            curve: metricSeries.curve ?? 'linear',
            dotRadius: 0,
        };
    }) as [LineChartSeriesConfig, ...LineChartSeriesConfig[]];
};

const getTooltipStep = (payload: RuntimeTooltipEntry[]) => {
    for (const entry of payload) {
        const dataKey = entry.dataKey;
        const step = dataKey === undefined ? undefined : entry.payload?.stepsBySeries?.[dataKey];

        if (typeof step === 'number') {
            return step;
        }
    }

    return payload.find((entry) => typeof entry.payload?.step === 'number')?.payload?.step ?? undefined;
};

export const MetricTooltip = ({ active, label, payload, valueFormatter }: MetricTooltipProps) => {
    const theme = useChartsTheme();
    const tooltipPayload = payload as RuntimeTooltipEntry[] | undefined;

    if (!active || !tooltipPayload?.length) {
        return null;
    }

    const row = tooltipPayload.find((entry) => entry.payload !== undefined)?.payload;
    const tooltipLabel = formatTooltipLabel(row?.epoch ?? label, getTooltipStep(tooltipPayload));
    const containerStyle: CSSProperties = {
        backgroundColor: theme.tooltip.backgroundColor,
        border: `1px solid ${theme.tooltip.borderColor}`,
        borderRadius: theme.tooltip.borderRadius,
        color: theme.tooltip.color,
        padding: theme.tooltip.padding,
        boxShadow: theme.tooltip.boxShadow,
        fontSize: theme.typography.fontSize,
        fontFamily: theme.typography.fontFamily,
    };
    const labelStyle: CSSProperties = { marginBottom: 4, fontWeight: 600, color: theme.tooltip.color };
    const itemStyle: CSSProperties = { display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 };

    return (
        <div style={containerStyle}>
            <div style={labelStyle}>{tooltipLabel}</div>
            {tooltipPayload.map((entry, index) => (
                <div style={itemStyle} key={`tooltip-item-${index}`}>
                    <span
                        style={{
                            display: 'inline-block',
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            backgroundColor: entry.color,
                            flexShrink: 0,
                        }}
                    />
                    <span>
                        {entry.name}: <strong>{getFormattedValue(entry.value, valueFormatter)}</strong>
                        {entry.unit}
                    </span>
                </div>
            ))}
        </div>
    );
};

export const MetricChartBox = ({ title, children }: { title: string; children: ReactNode }) => {
    return (
        <Flex flex={1} direction={'column'} minWidth={'size-5000'}>
            <Box title={title} content={<View backgroundColor={'gray-50'} minHeight={'size-3000'}>{children}</View>} />
        </Flex>
    );
};
