import { ChartsThemeProvider } from '@geti-ui/charts';
import { Flex } from '@geti-ui/ui';

import { ActionPredictionErrorGraph } from './action-prediction-error-graph.component';
import { LearningRateGraph } from './learning-rate-graph.component';
import { hasAnyMetric, MetricsEntry } from './metrics-chart-utils';
import { ModelUpdateSizeGraph } from './model-update-size-graph.component';
import { SystemAcceleratorUtilizationGraph } from './system-accelerator-utilization-graph.component';
import { SystemMemoryGraph } from './system-memory-graph.component';
import { SystemStepPerEpochGraph } from './system-step-per-epoch-graph.component';
import { SystemStepTimeGraph } from './system-step-time-graph.component';
import { TrainingLossGraph } from './training-loss-graph.component';

export const MetricsGraphs = ({ data }: { data?: MetricsEntry[] }) => {
    const hasActionError =
        hasAnyMetric(data, 'train_action_error_step') ||
        hasAnyMetric(data, 'train_action_error_epoch') ||
        hasAnyMetric(data, 'val_action_error');
    const hasAcceleratorUtilization =
        hasAnyMetric(data, 'system_accelerator_utilization_percent') ||
        hasAnyMetric(data, 'system_accelerator_power_w');

    return (
        <ChartsThemeProvider theme={{ dotRadius: 0, activeDotRadius: 0 }}>
            <Flex direction='row' gap='size-200' wrap flex={1}>
                <TrainingLossGraph data={data} />
                {hasActionError && <ActionPredictionErrorGraph data={data} />}
                <LearningRateGraph data={data} />
                <ModelUpdateSizeGraph data={data} />
            </Flex>
            <Flex direction='row' gap='size-200' wrap flex={1}>
                <SystemMemoryGraph data={data} />
                {hasAcceleratorUtilization && <SystemAcceleratorUtilizationGraph data={data} />}
                <SystemStepTimeGraph data={data} />
                <SystemStepPerEpochGraph data={data} />
            </Flex>
        </ChartsThemeProvider>
    );
};
