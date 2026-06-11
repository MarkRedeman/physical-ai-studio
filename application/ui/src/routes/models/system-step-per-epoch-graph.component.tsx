import { LineChart } from '@geti-ui/charts';

import {
    buildChartData,
    buildChartSeries,
    CHART_HEIGHT,
    CHART_MARGIN,
    getFormattedValue,
    MetricChartBox,
    MetricChartPoint,
    MetricTooltip,
    MetricsEntry,
    TRAIN_COLOR,
    X_AXIS_TICK_COUNT,
    Y_AXIS_TICK_COUNT,
} from './metrics-chart-utils';

const formatIntegerTick = (value: number) => Math.floor(value).toString();

const toStepEpochPoints = (data: MetricsEntry[] | undefined) => {
    if (!data) {
        return [];
    }

    return data.flatMap((entry): MetricChartPoint[] => {
        if (typeof entry.step !== 'number' || typeof entry.train_fractional_epoch !== 'number') {
            return [];
        }

        return [
            {
                x: entry.step,
                y: Math.floor(entry.train_fractional_epoch),
                epoch: entry.train_fractional_epoch,
                step: entry.step,
            },
        ];
    });
};

export const SystemStepPerEpochGraph = ({ data }: { data?: MetricsEntry[] }) => {
    const series = [
        {
            dataKey: 'epoch',
            name: 'Epoch',
            data: toStepEpochPoints(data),
            color: TRAIN_COLOR,
            curve: 'stepAfter' as const,
        },
    ];

    return (
        <MetricChartBox title='Steps Per Epoch'>
            <LineChart
                data={buildChartData(series, false)}
                xAxisKey='epoch'
                series={buildChartSeries(series)}
                showLegend={false}
                aria-label='Steps Per Epoch over Step'
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
                    label: { value: 'Step', position: 'bottom', fill: '#666', offset: 12 },
                    tickCount: X_AXIS_TICK_COUNT,
                    tickMargin: 12,
                    tickFormatter: (value) => formatIntegerTick(Number(value)),
                }}
                yAxisProps={{
                    label: { value: 'Epoch', angle: -90, position: 'center', dx: -38, fill: '#666' },
                    tickCount: Y_AXIS_TICK_COUNT,
                    tickMargin: 12,
                    tickFormatter: (value) => formatIntegerTick(Number(value)),
                }}
            />
        </MetricChartBox>
    );
};
