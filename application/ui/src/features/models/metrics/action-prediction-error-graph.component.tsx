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
    MetricTooltip,
    MetricsEntry,
    smooth,
    STEP_X_KEY,
    toPoints,
    TRAIN_EPOCH_X_KEY,
    VAL_EPOCH_X_KEY,
    Y_AXIS_TICK_COUNT,
} from './metrics-chart-utils';

const buildActionErrorSeries = (data?: MetricsEntry[]) => {
    return withDatasetSubsetPalette(
        [
            {
                dataKey: 'train',
                name: 'Training error',
                data: downsamplePointsByX(smooth(toPoints(data, STEP_X_KEY, 'train_action_error_step'))),
            },
            {
                dataKey: 'trainEpoch',
                name: 'Epoch average',
                data: toPoints(data, TRAIN_EPOCH_X_KEY, 'train_action_error_epoch'),
                dashed: true,
            },
            {
                dataKey: 'val',
                name: 'Validation error',
                data: toPoints(data, VAL_EPOCH_X_KEY, 'val_action_error'),
            },
        ],
        { matchBy: 'dataKey', aliases: { train: ['train', 'trainEpoch'], validation: ['val'] } }
    ).filter(hasData);
};

type ActionPredictionErrorGraphProps = {
    data?: MetricsEntry[];
    epochTicks: number[];
};

export const ActionPredictionErrorGraph = ({ data, epochTicks }: ActionPredictionErrorGraphProps) => {
    const series = useMemo(() => buildActionErrorSeries(data), [data]);
    const chartData = useMemo(() => buildChartData(series), [series]);
    const chartSeries = useMemo(() => buildChartSeries(series), [series]);

    return (
        <MetricChartBox title='Action Prediction Error'>
            <LineChart
                data={chartData}
                xAxisKey='epoch'
                series={chartSeries}
                showLegend={series.length > 1}
                aria-label='Action Prediction Error over Epoch'
                height={CHART_HEIGHT}
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
                    label: { value: 'Mean absolute error', angle: -90, position: 'center', dx: -38, fill: '#666' },
                    tickCount: Y_AXIS_TICK_COUNT,
                    tickMargin: 12,
                    tickFormatter: (value) => formatAxisValue(Number(value)),
                }}
            />
        </MetricChartBox>
    );
};
