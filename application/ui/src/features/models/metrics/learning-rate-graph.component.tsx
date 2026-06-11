import { LineChart } from '@geti-ui/charts';

import {
    buildChartData,
    buildChartSeries,
    CHART_HEIGHT,
    CHART_MARGIN,
    formatEpochTick,
    formatScientific,
    getNaturalEpochTicks,
    getFormattedValue,
    MetricChartBox,
    MetricChartPoint,
    MetricTooltip,
    MetricsEntry,
    STEP_X_KEY,
    toPoints,
    TRAIN_COLOR,
    Y_AXIS_TICK_COUNT,
} from './metrics-chart-utils';

const getRelativePaddedDomain = (points: MetricChartPoint[], paddingRatio = 0.1): [number, number] | undefined => {
    if (points.length === 0) {
        return undefined;
    }

    const values = points.map(({ y }) => y);
    const min = Math.min(...values);
    const max = Math.max(...values);

    return [min * (1 - paddingRatio), max * (1 + paddingRatio)];
};

export const LearningRateGraph = ({ data }: { data?: MetricsEntry[] }) => {
    const series = [
        {
            dataKey: 'lr',
            name: 'Learning rate',
            data: toPoints(data, STEP_X_KEY, 'train_lr'),
            color: TRAIN_COLOR,
        },
    ];
    const lrDomain = getRelativePaddedDomain(series[0].data);
    const epochTicks = getNaturalEpochTicks(data);

    return (
        <MetricChartBox title='Learning Rate'>
            <LineChart
                data={buildChartData(series)}
                xAxisKey='epoch'
                series={buildChartSeries(series)}
                showLegend={false}
                aria-label='Learning Rate over Epoch'
                height={CHART_HEIGHT}
                yScale={lrDomain === undefined ? undefined : { domain: lrDomain }}
                margin={CHART_MARGIN}
                highlight={{ enabled: true, interaction: { legendHover: true, legendClick: true } }}
                tooltipProps={{
                    formatter: (value, name) => [getFormattedValue(value, formatScientific), name],
                    content: (props) => <MetricTooltip {...props} valueFormatter={formatScientific} />,
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
                    label: { value: 'Learning rate', angle: -90, position: 'center', dx: -38, fill: '#666' },
                    tickCount: Y_AXIS_TICK_COUNT,
                    tickMargin: 12,
                    tickFormatter: (value) => formatScientific(Number(value)),
                }}
            />
        </MetricChartBox>
    );
};
