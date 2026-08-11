import { describe, expect, it } from 'vitest';

import { filterLossStepMetrics, filterLrMetrics, filterValLossMetrics, type MetricsEntry } from './metrics';

const entry = (partial: Partial<MetricsEntry>): MetricsEntry => ({
    epoch: 0,
    step: 0,
    train_loss: null,
    train_loss_step: null,
    val_loss: null,
    'lr-AdamW': null,
    ...partial,
});

describe('filterLossStepMetrics', () => {
    it('returns an empty array for no data', () => {
        expect(filterLossStepMetrics(undefined)).toEqual([]);
        expect(filterLossStepMetrics([])).toEqual([]);
    });

    it('prefers train_loss over train_loss_step', () => {
        const points = filterLossStepMetrics([entry({ step: 10, train_loss: 0.5, train_loss_step: 0.9 })]);
        expect(points).toEqual([{ x: 10, y: 0.5 }]);
    });

    it('falls back to train_loss_step when train_loss is missing', () => {
        const points = filterLossStepMetrics([entry({ step: 20, train_loss_step: 0.8 })]);
        expect(points).toEqual([{ x: 20, y: 0.8 }]);
    });

    it('skips rows with no loss value', () => {
        const points = filterLossStepMetrics([
            entry({ step: 1, train_loss: null, train_loss_step: null }),
            entry({ step: 2, train_loss: 0.3 }),
        ]);
        expect(points).toEqual([{ x: 2, y: 0.3 }]);
    });
});

describe('filterValLossMetrics', () => {
    it('keeps rows with a validation loss', () => {
        const points = filterValLossMetrics([entry({ step: 5, val_loss: null }), entry({ step: 6, val_loss: 0.25 })]);
        expect(points).toEqual([{ x: 6, y: 0.25 }]);
    });
});

describe('filterLrMetrics', () => {
    it('keeps rows with a learning rate', () => {
        const points = filterLrMetrics([entry({ step: 5, 'lr-AdamW': null }), entry({ step: 6, 'lr-AdamW': 0.0001 })]);
        expect(points).toEqual([{ x: 6, y: 0.0001 }]);
    });
});
