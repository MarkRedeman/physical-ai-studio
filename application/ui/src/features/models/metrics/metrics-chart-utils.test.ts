import { compressStepEpochPoints, downsamplePointsByX, type MetricChartPoint } from './metrics-chart-utils';

describe('metrics chart downsampling', () => {
    it('returns the same array reference when under max points', () => {
        const points: MetricChartPoint[] = [
            { x: 0, y: 1, epoch: 0, step: 0 },
            { x: 1, y: 2, epoch: 1, step: 1 },
            { x: 2, y: 3, epoch: 2, step: 2 },
        ];

        const result = downsamplePointsByX(points, 10);

        expect(result).toBe(points);
    });

    it('downsamples by x while preserving first and last points', () => {
        const points: MetricChartPoint[] = Array.from({ length: 100 }, (_, index) => ({
            x: index,
            y: Math.sin(index / 10),
            epoch: index / 10,
            step: index,
        }));

        const result = downsamplePointsByX(points, 12);

        expect(result.length).toBeLessThanOrEqual(12);
        expect(result[0]).toEqual(points[0]);
        expect(result[result.length - 1]).toEqual(points[points.length - 1]);
        expect(result.every((point, index) => index === 0 || result[index - 1].x <= point.x)).toBe(true);
    });

    it('keeps endpoints when all x values are identical', () => {
        const points: MetricChartPoint[] = Array.from({ length: 20 }, (_, index) => ({
            x: 1,
            y: index,
            step: index,
        }));

        const result = downsamplePointsByX(points, 6);

        expect(result.length).toBe(6);
        expect(result[0]).toEqual(points[0]);
        expect(result[result.length - 1]).toEqual(points[points.length - 1]);
    });
});

describe('step-per-epoch compression', () => {
    it('keeps only transition points and endpoints', () => {
        const points: MetricChartPoint[] = [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 1 },
            { x: 4, y: 1 },
            { x: 5, y: 2 },
        ];

        const result = compressStepEpochPoints(points);

        expect(result.map((point) => point.x)).toEqual([0, 2, 3, 4, 5]);
        expect(result.map((point) => point.y)).toEqual([0, 0, 1, 1, 2]);
    });

    it('returns first and last point when epoch does not change', () => {
        const points: MetricChartPoint[] = [
            { x: 0, y: 1 },
            { x: 1, y: 1 },
            { x: 2, y: 1 },
            { x: 3, y: 1 },
        ];

        const result = compressStepEpochPoints(points);

        expect(result).toEqual([points[0], points[points.length - 1]]);
    });
});
