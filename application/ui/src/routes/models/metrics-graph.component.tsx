import { useId } from 'react';

import { Flex, View } from '@geti-ui/ui';
import {
    Area,
    AreaChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
    type TooltipContentProps,
} from 'recharts';

import { Box } from './box.component';

export type MetricGraphPoint = {
    x: number;
    [seriesKey: string]: number;
};

export type MetricSeries = {
    dataKey: string;
    name: string;
    color: string;
};

type MetricGraphProps = {
    title: string;
    data?: MetricGraphPoint[];
    xAxisLabel?: string;
    yAxisLabel: string;
    yTickFormatter?: (value: number) => string;
    series: MetricSeries[];
};

const X_AXIS_TICK_COUNT = 8;
const Y_AXIS_TICK_COUNT = 4;

const defaultTickFormatter = (value: number) => Number(value).toFixed(4);

const MetricChartTooltip = ({
    active,
    label,
    payload,
    valueFormatter,
}: TooltipContentProps & { valueFormatter: (value: number) => string }) => {
    if (!active || !payload?.length) {
        return null;
    }
    return (
        <div
            style={{
                backgroundColor: 'var(--spectrum-global-color-gray-100)',
                border: '1px solid var(--spectrum-global-color-gray-500)',
                borderRadius: 4,
                padding: '6px 10px',
                fontSize: 12,
                color: 'var(--spectrum-global-color-gray-800)',
            }}
        >
            <div>Step: {label}</div>
            {payload.map((entry) => (
                <div key={entry.name ?? String(entry.dataKey)}>
                    <span style={{ color: entry.color }}>
                        {entry.name}: {entry.value == null ? '—' : valueFormatter(Number(entry.value))}
                    </span>
                </div>
            ))}
        </div>
    );
};

export const MetricGraph = ({ title, data, xAxisLabel, yAxisLabel, yTickFormatter, series }: MetricGraphProps) => {
    const baseGradientId = useId();
    const valueFormatter = yTickFormatter ?? defaultTickFormatter;

    return (
        <Flex flex={1} direction={'column'} minWidth={'size-5000'}>
            <Box
                title={title}
                content={
                    <View backgroundColor={'gray-50'} minHeight={'size-3000'}>
                        <ResponsiveContainer width='100%' height={300} style={{ userSelect: 'none' }}>
                            <AreaChart
                                style={{ aspectRatio: 1.6 }}
                                data={data}
                                margin={{ top: 35, bottom: 35, left: 35 }}
                            >
                                <defs>
                                    {series.map((metricSeries, index) => (
                                        <linearGradient
                                            key={metricSeries.dataKey}
                                            id={`${baseGradientId}-${index}`}
                                            x1='0'
                                            y1='0'
                                            x2='0'
                                            y2='1'
                                        >
                                            <stop offset='5%' stopColor={metricSeries.color} stopOpacity={0.3} />
                                            <stop offset='95%' stopColor={metricSeries.color} stopOpacity={0} />
                                        </linearGradient>
                                    ))}
                                </defs>
                                <CartesianGrid />
                                <XAxis
                                    dataKey='x'
                                    type='number'
                                    domain={['dataMin', 'dataMax']}
                                    label={{ value: xAxisLabel ?? 'x', position: 'bottom', fill: '#666', offset: 12 }}
                                    tickCount={X_AXIS_TICK_COUNT}
                                    tickMargin={12}
                                />
                                <YAxis
                                    label={{ value: yAxisLabel, angle: -90, position: 'center', dx: -38, fill: '#666' }}
                                    tickCount={Y_AXIS_TICK_COUNT}
                                    tickMargin={12}
                                    tickFormatter={(value) => valueFormatter(Number(value))}
                                />
                                {series.map((metricSeries, index) => (
                                    <Area
                                        key={metricSeries.dataKey}
                                        type='linear'
                                        dataKey={metricSeries.dataKey}
                                        name={metricSeries.name}
                                        stroke={metricSeries.color}
                                        strokeWidth={2}
                                        fill={`url(#${baseGradientId}-${index})`}
                                        dot={false}
                                        isAnimationActive={false}
                                        connectNulls
                                    />
                                ))}
                                <Tooltip
                                    content={(props) => (
                                        <MetricChartTooltip {...props} valueFormatter={valueFormatter} />
                                    )}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </View>
                }
            />
        </Flex>
    );
};
