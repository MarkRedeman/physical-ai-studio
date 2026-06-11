import { useMemo } from 'react';

import { ChartsThemeProvider } from '@geti-ui/charts';
import { Flex } from '@geti-ui/ui';

import { ActionPredictionErrorGraph } from './action-prediction-error-graph.component';
import { LearningRateGraph } from './learning-rate-graph.component';
import { CHART_THEME, getEquidistantEpochTicks, MetricsEntry } from './metrics-chart-utils';
import { ModelUpdateSizeGraph } from './model-update-size-graph.component';
import { SystemAcceleratorUtilizationGraph } from './system-accelerator-utilization-graph.component';
import { SystemMemoryGraph } from './system-memory-graph.component';
import { SystemStepPerEpochGraph } from './system-step-per-epoch-graph.component';
import { SystemStepTimeGraph } from './system-step-time-graph.component';
import { TrainingLossGraph } from './training-loss-graph.component';

export const MetricsGraphs = ({ data }: { data?: MetricsEntry[] }) => {
    const epochTicks = useMemo(() => getEquidistantEpochTicks(data), [data]);
    const { hasActionError, hasAcceleratorUtilization } = useMemo(() => {
        let hasActionErrorMetric = false;
        let hasAcceleratorUtilizationMetric = false;

        for (const entry of data ?? []) {
            if (!hasActionErrorMetric) {
                hasActionErrorMetric =
                    typeof entry.train_action_error_step === 'number' ||
                    typeof entry.train_action_error_epoch === 'number' ||
                    typeof entry.val_action_error === 'number';
            }

            if (!hasAcceleratorUtilizationMetric) {
                hasAcceleratorUtilizationMetric =
                    typeof entry.system_accelerator_utilization_percent === 'number' ||
                    typeof entry.system_accelerator_power_w === 'number';
            }

            if (hasActionErrorMetric && hasAcceleratorUtilizationMetric) {
                break;
            }
        }

        return {
            hasActionError: hasActionErrorMetric,
            hasAcceleratorUtilization: hasAcceleratorUtilizationMetric,
        };
    }, [data]);

    return (
        <ChartsThemeProvider theme={CHART_THEME}>
            <Flex direction='row' gap='size-200' wrap flex={1}>
                <TrainingLossGraph data={data} epochTicks={epochTicks} />
                {hasActionError && <ActionPredictionErrorGraph data={data} epochTicks={epochTicks} />}
                <LearningRateGraph data={data} epochTicks={epochTicks} />
                <ModelUpdateSizeGraph data={data} epochTicks={epochTicks} />
            </Flex>
            <Flex direction='row' gap='size-200' wrap flex={1}>
                <SystemMemoryGraph data={data} epochTicks={epochTicks} />
                {hasAcceleratorUtilization && <SystemAcceleratorUtilizationGraph data={data} epochTicks={epochTicks} />}
                <SystemStepTimeGraph data={data} epochTicks={epochTicks} />
                <SystemStepPerEpochGraph data={data} />
            </Flex>
        </ChartsThemeProvider>
    );
};
