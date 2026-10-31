import { screen } from '@testing-library/react';

import { getMockedTrainJobPayload } from '../../../test-utils/mocks/mock-train-job-payload';
import { render } from '../../../test-utils/render';
import { TrainingParameters } from './training-configuration';

describe('Training configuration', () => {
    it('displays the step budget and step-based distillation', () => {
        const summary = getMockedTrainJobPayload({
            max_epochs: null,
            max_steps: 100,
            snapflow_enabled: true,
            snapflow_distill_epochs: null,
            snapflow_distill_steps: 25,
        });
        render(<TrainingParameters summary={summary} />);
        expect(screen.getByText('Max steps')).toBeInTheDocument();
        expect(screen.getByText('Distillation steps')).toBeInTheDocument();
        expect(screen.getByText('100')).toBeInTheDocument();
        expect(screen.getByText('25')).toBeInTheDocument();
        expect(screen.queryByText('Max epochs')).not.toBeInTheDocument();
        expect(screen.queryByText('Distillation epochs')).not.toBeInTheDocument();
    });
});
