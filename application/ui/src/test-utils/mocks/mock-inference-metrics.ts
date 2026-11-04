import { SchemaInferenceMetricsMessage } from '../../api/openapi-spec';

export const getMockedInferenceMetrics = (
    overrides: Partial<SchemaInferenceMetricsMessage> = {}
): SchemaInferenceMetricsMessage => ({
    type: 'inference_metrics',
    model_id: 'model-1',
    backend: 'onnx',
    device: 'cpu',
    run_id: 'run-1',
    timestamp: 1000,
    latency_ms: 42,
    average_latency_ms: 38,
    sample_count: 20,
    prediction_count: 21,
    is_warmup: false,
    ...overrides,
});
