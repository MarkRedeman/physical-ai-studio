import { describe, expect, it } from '@rstest/core';

import { getWizardSteps } from './wizard-steps';

describe('getWizardSteps', () => {
    it('uses the configured camera slots to include feature mapping', () => {
        expect(getWizardSteps(true)).toContain('feature-mapping');
        expect(getWizardSteps(false)).not.toContain('feature-mapping');
    });

    it('skips camera mapping when continuing from a checkpoint', () => {
        expect(getWizardSteps(true, true)).not.toContain('feature-mapping');
    });
});
