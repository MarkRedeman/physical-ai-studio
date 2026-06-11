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

type SystemAcceleratorUtilizationGraphProps = {
    data?: MetricsEntry[];
    epochTicks: number[];
};

export const SystemAcceleratorUtilizationGraph = ({
    data,
    epochTicks,
}: SystemAcceleratorUtilizationGraphProps) => {
    const utilizationSeries = useMemo<MetricChartSeries>(
        () => ({
            dataKey: 'utilization',
            name: 'Accelerator utilization',
            data: toPoints(data, STEP_X_KEY, 'system_accelerator_utilization_percent'),
            color: TRAIN_COLOR,
        }),
        [data]
    );
    const powerSeries = useMemo<MetricChartSeries>(
        () => ({
            dataKey: 'power',
            name: 'Accelerator power',
            data: toPoints(data, STEP_X_KEY, 'system_accelerator_power_w'),
            color: TRAIN_COLOR,
        }),
        [data]
    );
    const isUtilization = hasSeriesData(utilizationSeries);
    const series = useMemo(() => (isUtilization ? [utilizationSeries] : [powerSeries]), [isUtilization, powerSeries, utilizationSeries]);
    const chartData = useMemo(() => buildChartData(series), [series]);
    const chartSeries = useMemo(() => buildChartSeries(series), [series]);

    return (
        <MetricChartBox title={isUtilization ? 'Accelerator Utilization' : 'Accelerator Power'}>
            <LineChart
                data={chartData}
                xAxisKey='epoch'
                series={chartSeries}
                showLegend={false}
                aria-label={`${isUtilization ? 'Accelerator Utilization' : 'Accelerator Power'} over Epoch`}
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
                    label: {
                        value: isUtilization ? '%' : 'Watts',
                        angle: -90,
                        position: 'center',
                        dx: -38,
                        fill: '#666',
                    },
                    tickCount: Y_AXIS_TICK_COUNT,
                    tickMargin: 12,
                    tickFormatter: (value) => formatAxisValue(Number(value)),
                }}
            />
        </MetricChartBox>
    );
};
