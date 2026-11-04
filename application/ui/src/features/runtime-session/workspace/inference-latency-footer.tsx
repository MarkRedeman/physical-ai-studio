import { useEffect, useState, useSyncExternalStore } from 'react';

import { ActionButton, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { InfoOutline } from '@geti-ui/ui/icons';

import { failureOf, useRuntimeSession } from '../runtime-session-provider';

import classes from './inference-latency-footer.module.css';

const milliseconds = (value: number) => `${value < 10 ? value.toFixed(1) : Math.round(value)} ms`;

export const InferenceLatencyFooter = () => {
    const { state, statuses, inferenceMetrics } = useRuntimeSession();
    const sample = useSyncExternalStore(
        (listener) => inferenceMetrics?.subscribe(listener) ?? (() => {}),
        () => inferenceMetrics?.value ?? null,
        () => null
    );
    const hasFailure = failureOf(statuses, 'model') !== null || failureOf(statuses, 'control') !== null;
    const isRunning = state?.control?.type === 'model' && !hasFailure;
    const [now, setNow] = useState(Date.now);
    useEffect(() => {
        if (!isRunning) return;
        const timer = window.setInterval(() => setNow(Date.now()), 1000);
        return () => window.clearInterval(timer);
    }, [isRunning]);

    const metrics =
        isRunning &&
        sample != null &&
        sample.data.model_id === state?.model?.model_id &&
        sample.data.backend === state?.model?.backend &&
        sample.data.device === state?.model?.device
            ? sample.data
            : null;
    const ageSeconds = sample == null ? 0 : Math.max(0, Math.floor((now - sample.receivedAt) / 1000));

    return (
        <footer className={classes.footer} aria-label='Inference performance'>
            <div className={classes.values}>
                {metrics == null ? (
                    <span className={classes.label}>
                        {hasFailure
                            ? 'Inference timing unavailable'
                            : isRunning
                              ? 'Waiting for first prediction…'
                              : 'Run the model to see latency'}
                    </span>
                ) : (
                    <>
                        <span className={classes.label}>
                            {metrics.is_warmup ? 'Warm-up' : ageSeconds >= 3 ? 'Last result' : 'Latency'}
                        </span>
                        <strong className={classes.latest}>{milliseconds(metrics.latency_ms)}</strong>
                        {metrics.average_latency_ms != null && (
                            <span className={classes.average}>avg {milliseconds(metrics.average_latency_ms)}</span>
                        )}
                    </>
                )}
            </div>
            <TooltipTrigger delay={300}>
                <ActionButton isQuiet aria-label='About inference latency' UNSAFE_className={classes.help}>
                    <InfoOutline />
                </ActionButton>
                <Tooltip>
                    Model prediction time per action chunk, including model preprocessing and excluding worker
                    communication. The average uses up to 20 recent predictions and excludes the first warm-up
                    prediction of each run.
                    {metrics != null &&
                        ` Last result received ${ageSeconds}s ago; average of ${metrics.sample_count} predictions.`}
                </Tooltip>
            </TooltipTrigger>
        </footer>
    );
};
