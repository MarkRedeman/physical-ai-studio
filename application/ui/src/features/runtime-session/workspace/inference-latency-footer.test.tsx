import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';

import { getMockedInferenceMetrics } from '../../../test-utils/mocks/mock-inference-metrics';
import { getMockedRuntimeSession } from '../../../test-utils/mocks/mock-runtime-session';
import { render } from '../../../test-utils/render';
import { InferenceMetricsSample, Statuses } from '../runtime-session-provider';
import { InferenceLatencyFooter } from './inference-latency-footer';

let state = getMockedRuntimeSession();
let statuses: Statuses = {};
const inferenceMetrics: { value: InferenceMetricsSample | null; subscribe: () => () => void } = {
    value: null,
    subscribe: () => () => {},
};
rs.mock('../runtime-session-provider', () => ({
    useRuntimeSession: () => ({ state, statuses, inferenceMetrics }),
    failureOf: (values: Statuses, part: string) => (values[part]?.state === 'failed' ? values[part].error : null),
}));

beforeEach(() => {
    statuses = {};
    state = getMockedRuntimeSession({
        model: { model_id: 'model-1', name: 'Cube policy', backend: 'onnx', device: 'cpu' },
        control: {
            type: 'model',
            task: 'Pick',
            hz: 30,
            grace_s: 3,
            execution: { mode: 'sync', request_threshold: 0.5 },
        },
    });
    inferenceMetrics.value = { data: getMockedInferenceMetrics(), receivedAt: Date.now() };
});

it('shows the latest prediction and recent average', () => {
    render(<InferenceLatencyFooter />);
    expect(screen.getByText('42 ms')).toBeInTheDocument();
    expect(screen.getByText('avg 38 ms')).toBeInTheDocument();
});

it('shows a waiting state before the first prediction', () => {
    inferenceMetrics.value = null;
    render(<InferenceLatencyFooter />);
    expect(screen.getByText('Waiting for first prediction…')).toBeInTheDocument();
});

it('does not show old measurements after stopping', () => {
    state.control = null;
    render(<InferenceLatencyFooter />);
    expect(screen.getByText('Run the model to see latency')).toBeInTheDocument();
    expect(screen.queryByText('42 ms')).not.toBeInTheDocument();
});

it('distinguishes warmup and excludes its average', () => {
    inferenceMetrics.value = {
        data: getMockedInferenceMetrics({
            is_warmup: true,
            latency_ms: 284,
            average_latency_ms: null,
            sample_count: 0,
            prediction_count: 1,
        }),
        receivedAt: Date.now(),
    };
    render(<InferenceLatencyFooter />);
    expect(screen.getByText('Warm-up')).toBeInTheDocument();
    expect(screen.getByText('284 ms')).toBeInTheDocument();
    expect(screen.queryByText(/avg/)).not.toBeInTheDocument();
});

it('does not reuse timing from another backend', () => {
    inferenceMetrics.value = { data: getMockedInferenceMetrics({ backend: 'openvino' }), receivedAt: Date.now() };
    render(<InferenceLatencyFooter />);
    expect(screen.getByText('Waiting for first prediction…')).toBeInTheDocument();
});

it('labels a result that has not updated recently', () => {
    inferenceMetrics.value = { data: getMockedInferenceMetrics(), receivedAt: Date.now() - 5000 };
    render(<InferenceLatencyFooter />);
    expect(screen.getByText('Last result')).toBeInTheDocument();
});
