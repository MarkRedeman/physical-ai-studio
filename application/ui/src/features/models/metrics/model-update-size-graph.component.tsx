import { useMemo } from 'react';

import { LineChart } from '@geti-ui/charts';

import {
    buildChartData,
    buildChartSeries,
    CHART_HIGHLIGHT,
    CHART_HEIGHT,
    CHART_MARGIN,
    formatAxisValue,
    formatEpochTick,
    getFormattedValue,
    MetricChartBox,
    MetricTooltip,
    MetricsEntry,
    STEP_X_KEY,
    toPoints,
    TRAIN_COLOR,
    Y_AXIS_TICK_COUNT,
} from './metrics-chart-utils';

type ModelUpdateSizeGraphProps = {
    data?: MetricsEntry[];
    epochTicks: number[];
};

export const ModelUpdateSizeGraph = ({ data, epochTicks }: ModelUpdateSizeGraphProps) => {
    const series = useMemo(
        () => [
            {
                dataKey: 'gradNorm',
                name: 'Model update size',
                data: toPoints(data, STEP_X_KEY, 'train_grad_norm'),
                color: TRAIN_COLOR,
            },
        ],
        [data]
    );
    const chartData = useMemo(() => buildChartData(series), [series]);
    const chartSeries = useMemo(() => buildChartSeries(series), [series]);

    return (
        <MetricChartBox title='Model Update Size'>
            <LineChart
                data={chartData}
                xAxisKey='epoch'
                series={chartSeries}
                showLegend={false}
                aria-label='Model Update Size over Epoch'
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
                    label: { value: 'Gradient norm', angle: -90, position: 'center', dx: -38, fill: '#666' },
                    tickCount: Y_AXIS_TICK_COUNT,
                    tickMargin: 12,
                    tickFormatter: (value) => formatAxisValue(Number(value)),
                }}
            />
        </MetricChartBox>
    );
};
