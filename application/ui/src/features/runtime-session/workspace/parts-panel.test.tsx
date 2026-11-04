import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { getMockedRuntimeSession } from '../../../test-utils/mocks/mock-runtime-session';
import { render } from '../../../test-utils/render';
import { Statuses } from '../runtime-session-provider';
import { EnvironmentRow, PartActions } from './parts-panel';

const healthy: Statuses = {
    environment: { type: 'status', component: 'environment', state: 'running' },
    'environment/follower': { type: 'status', component: 'environment/follower', state: 'running' },
};
const failed: Statuses = {
    ...healthy,
    'environment/leader': { type: 'status', component: 'environment/leader', state: 'failed', error: 'Unplugged' },
};
const environment = getMockedRuntimeSession().environment;

describe('EnvironmentRow', () => {
    it('keeps healthy device details collapsed', () => {
        render(<EnvironmentRow environment={environment} statuses={healthy} isPending={false} />);
        expect(screen.getByText('Workbench')).toBeInTheDocument();
        expect(screen.getByText('Follower')).not.toBeVisible();
    });

    it('shows failed device details when expanded', () => {
        render(<EnvironmentRow environment={environment} statuses={failed} isPending={false} showDetails />);
        expect(screen.getByText('Leader')).toBeInTheDocument();
        expect(screen.getByText('Unplugged')).toBeInTheDocument();
    });

    it('shows and hides details using the header-controlled state', () => {
        const { rerender } = render(<EnvironmentRow environment={environment} statuses={healthy} isPending={false} />);
        expect(screen.getByText('Follower')).not.toBeVisible();
        rerender(<EnvironmentRow environment={environment} statuses={failed} isPending={false} showDetails />);
        expect(screen.getByText('Unplugged')).toBeVisible();
    });
});

describe('PartActions', () => {
    it('offers changing a healthy resource without developer-oriented actions', async () => {
        const user = userEvent.setup();
        render(
            <PartActions
                part='dataset'
                isLoaded
                isPending={false}
                isDisabled={false}
                canChoose
                canRetry={false}
                onAction={rs.fn()}
            />
        );
        await user.click(screen.getByRole('button', { name: 'dataset actions' }));
        expect(screen.getByRole('menuitem', { name: 'Change dataset…' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: /Reload|Unload|Try again/ })).not.toBeInTheDocument();
    });

    it('offers reconnecting devices when an environment fails', async () => {
        const user = userEvent.setup();
        const onAction = rs.fn();
        render(
            <PartActions
                part='environment'
                isLoaded
                isPending={false}
                isDisabled={false}
                canChoose
                canRetry
                onAction={onAction}
            />
        );
        await user.click(screen.getByRole('button', { name: 'environment actions' }));
        await user.click(screen.getByRole('menuitem', { name: 'Reconnect devices' }));
        expect(onAction).toHaveBeenCalledWith('reload');
    });
});
