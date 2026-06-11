import { LineChart } from '@geti-ui/charts';

import {
    buildChartData,
    buildChartSeries,
    CHART_HEIGHT,
    CHART_MARGIN,
    formatAxisValue,
    formatEpochTick,
    getNaturalEpochTicks,
    getFormattedValue,
    MetricChartBox,
    MetricTooltip,
    MetricsEntry,
    STEP_X_KEY,
    toPoints,
    TRAIN_COLOR,
    Y_AXIS_TICK_COUNT,
} from './metrics-chart-utils';

export const SystemStepTimeGraph = ({ data }: { data?: MetricsEntry[] }) => {
    const series = [
        {
            dataKey: 'stepTime',
            name: 'Step time',
            data: toPoints(data, STEP_X_KEY, 'system_step_time_s'),
            color: TRAIN_COLOR,
        },
    ];
    const epochTicks = getNaturalEpochTicks(data);

    return (
        <MetricChartBox title='Step Time'>
            <LineChart
                data={buildChartData(series)}
                xAxisKey='epoch'
                series={buildChartSeries(series)}
                showLegend={false}
                aria-label='Step Time over Epoch'
                height={CHART_HEIGHT}
                margin={CHART_MARGIN}
                highlight={{ enabled: true, interaction: { legendHover: true, legendClick: true } }}
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
                    label: { value: 'Seconds', angle: -90, position: 'center', dx: -38, fill: '#666' },
                    tickCount: Y_AXIS_TICK_COUNT,
                    tickMargin: 12,
                    tickFormatter: (value) => formatAxisValue(Number(value)),
                }}
            />
        </MetricChartBox>
    );
};
