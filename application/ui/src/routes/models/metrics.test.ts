import { describe, expect, it } from 'vitest';

import { buildLossMetrics, filterLrMetrics, type MetricsEntry } from './metrics';

const entry = (partial: Partial<MetricsEntry>): MetricsEntry => ({
    epoch: 0,
    step: 0,
    train_loss: null,
    train_loss_step: null,
    val_loss: null,
    'lr-AdamW': null,
    ...partial,
});

describe('buildLossMetrics', () => {
    it('returns an empty array for no data', () => {
        expect(buildLossMetrics(undefined)).toEqual([]);
        expect(buildLossMetrics([])).toEqual([]);
    });

    it('merges train and validation loss at the same step into one row', () => {
        const points = buildLossMetrics([entry({ step: 10, train_loss: 0.5 }), entry({ step: 10, val_loss: 0.3 })]);
        expect(points).toEqual([{ x: 10, train: 0.5, val: 0.3 }]);
    });

    it('prefers train_loss over train_loss_step', () => {
        const points = buildLossMetrics([entry({ step: 10, train_loss: 0.5, train_loss_step: 0.9 })]);
        expect(points).toEqual([{ x: 10, train: 0.5 }]);
    });

    it('falls back to train_loss_step when train_loss is missing', () => {
        const points = buildLossMetrics([entry({ step: 20, train_loss_step: 0.8 })]);
        expect(points).toEqual([{ x: 20, train: 0.8 }]);
    });

    it('skips rows with no loss value and sorts by step', () => {
        const points = buildLossMetrics([
            entry({ step: 30, train_loss: null, train_loss_step: null }),
            entry({ step: 1, train_loss: 0.3 }),
            entry({ step: 2, val_loss: 0.25 }),
        ]);
        expect(points).toEqual([
            { x: 1, train: 0.3 },
            { x: 2, val: 0.25 },
        ]);
    });
});

describe('filterLrMetrics', () => {
    it('keeps rows with a learning rate', () => {
        const points = filterLrMetrics([entry({ step: 5, 'lr-AdamW': null }), entry({ step: 6, 'lr-AdamW': 0.0001 })]);
        expect(points).toEqual([{ x: 6, lr: 0.0001 }]);
    });
});
