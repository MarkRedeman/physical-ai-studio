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
    MetricChartSeries,
    MetricTooltip,
    MetricsEntry,
    STEP_X_KEY,
    toPoints,
    TRAIN_COLOR,
    Y_AXIS_TICK_COUNT,
} from './metrics-chart-utils';

const hasSeriesData = (series: MetricChartSeries) => series.data.length > 0;

type SystemMemoryGraphProps = {
    data?: MetricsEntry[];
    epochTicks: number[];
};

export const SystemMemoryGraph = ({ data, epochTicks }: SystemMemoryGraphProps) => {
    const percentSeries = useMemo<MetricChartSeries>(
        () => ({
            dataKey: 'memoryPercent',
            name: 'Accelerator memory',
            data: toPoints(data, STEP_X_KEY, 'system_accelerator_memory_percent'),
            color: TRAIN_COLOR,
        }),
        [data]
    );
    const mbSeries = useMemo<MetricChartSeries>(
        () => ({
            dataKey: 'memory',
            name: 'Accelerator memory',
            data: toPoints(data, STEP_X_KEY, 'system_accelerator_memory_mb'),
            color: TRAIN_COLOR,
        }),
        [data]
    );
    const totalMbSeries = useMemo<MetricChartSeries>(
        () => ({
            dataKey: 'memoryTotal',
            name: 'Accelerator memory total',
            data: toPoints(data, STEP_X_KEY, 'system_accelerator_memory_total_mb'),
            color: '#ff7300',
            dashed: true,
        }),
        [data]
    );
    const isPercent = hasSeriesData(percentSeries);
    const series = useMemo(
        () => (isPercent ? [percentSeries] : [mbSeries, totalMbSeries].filter(hasSeriesData)),
        [isPercent, mbSeries, percentSeries, totalMbSeries]
    );
    const chartData = useMemo(() => buildChartData(series), [series]);
    const chartSeries = useMemo(() => buildChartSeries(series), [series]);

    return (
        <MetricChartBox title='Accelerator Memory'>
            <LineChart
                data={chartData}
                xAxisKey='epoch'
                series={chartSeries}
                showLegend={series.length > 1}
                aria-label='Accelerator Memory over Epoch'
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
                    label: { value: isPercent ? '%' : 'MB', angle: -90, position: 'center', dx: -38, fill: '#666' },
                    tickCount: Y_AXIS_TICK_COUNT,
                    tickMargin: 12,
                    tickFormatter: (value) => formatAxisValue(Number(value)),
                }}
            />
        </MetricChartBox>
    );
};
