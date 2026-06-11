import { useMemo } from 'react';

import { LineChart, withDatasetSubsetPalette } from '@geti-ui/charts';

import {
    buildChartData,
    buildChartSeries,
    CHART_HIGHLIGHT,
    CHART_HEIGHT,
    CHART_MARGIN,
    downsamplePointsByX,
    formatAxisValue,
    formatEpochTick,
    getFormattedValue,
    hasData,
    MetricChartBox,
    MetricChartSeries,
    MetricTooltip,
    MetricsEntry,
    smooth,
    STEP_X_KEY,
    toPoints,
    VAL_EPOCH_X_KEY,
    Y_AXIS_TICK_COUNT,
} from './metrics-chart-utils';

const LOG_LOSS_FLOOR = 1e-6;

const toPositiveLogPoints = (series: MetricChartSeries) => {
    return { ...series, data: series.data.map((point) => ({ ...point, y: Math.max(point.y, LOG_LOSS_FLOOR) })) };
};

const buildLossSeries = (data?: MetricsEntry[]) => {
    const series = withDatasetSubsetPalette(
        [
            {
                dataKey: 'train',
                name: 'Training loss',
                data: downsamplePointsByX(smooth(toPoints(data, STEP_X_KEY, 'train_loss_step'))),
            },
            {
                dataKey: 'val',
                name: 'Validation loss',
                data: toPoints(data, VAL_EPOCH_X_KEY, 'val_loss'),
            },
        ],
        { matchBy: 'dataKey', aliases: { train: ['train'], validation: ['val'] } }
    );

    return series.map(toPositiveLogPoints).filter(hasData);
};

type TrainingLossGraphProps = {
    data?: MetricsEntry[];
    epochTicks: number[];
};

export const TrainingLossGraph = ({ data, epochTicks }: TrainingLossGraphProps) => {
    const series = useMemo(() => buildLossSeries(data), [data]);
    const chartData = useMemo(() => buildChartData(series), [series]);
    const chartSeries = useMemo(() => buildChartSeries(series), [series]);

    return (
        <MetricChartBox title='Training Loss'>
            <LineChart
                data={chartData}
                xAxisKey='epoch'
                series={chartSeries}
                showLegend={series.length > 1}
                aria-label='Training Loss over Epoch'
                height={CHART_HEIGHT}
                yScale={{ scale: 'log', domain: ['auto', 'auto'] }}
                margin={CHART_MARGIN}
                highlight={CHART_HIGHLIGHT}
                tooltipProps={{
                    formatter: (value, name) => [getFormattedValue(value), name],
                    content: (props) => <MetricTooltip {...props} />,
                }}
                xAxisProps={{
                    type: 'number',
                    domain: ['dataMin', 'dataMax'],
                    label: { value: 'Epoch', position: 'bottom', fill: '#666', offset: 12 },
                    ticks: epochTicks,
                    tickMargin: 12,
                    tickFormatter: formatEpochTick,
                }}
                yAxisProps={{
                    label: { value: 'Loss', angle: -90, position: 'center', dx: -38, fill: '#666' },
                    tickCount: Y_AXIS_TICK_COUNT,
                    tickMargin: 12,
                    tickFormatter: (value) => formatAxisValue(Number(value)),
                }}
            />
        </MetricChartBox>
    );
};
