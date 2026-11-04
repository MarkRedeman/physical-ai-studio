import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedFollowerRobot, getMockedLeaderRobot } from '../../../test-utils/mocks/mock-robot';
import { getMockedSessionEnvironment } from '../../../test-utils/mocks/mock-runtime-session';
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
const environment = getMockedSessionEnvironment({ leader_id: 'leader-1' });
const options = { route: '/projects/project-1', path: '/projects/:project_id' };

beforeEach(() =>
    server.use(
        http.get('/api/projects/{project_id}/robots/{robot_id}', ({ params }) =>
            HttpResponse.json(params.robot_id === 'leader-1' ? getMockedLeaderRobot() : getMockedFollowerRobot())
        )
    )
);

describe('EnvironmentRow', () => {
    it('keeps healthy device details collapsed', () => {
        render(<EnvironmentRow environment={environment} statuses={healthy} isPending={false} />);
        expect(screen.getByText('Workbench')).toBeInTheDocument();
        expect(screen.queryByRole('group', { name: 'Follower robot' })).not.toBeInTheDocument();
    });

    it('shows failed device details when expanded', () => {
        render(<EnvironmentRow environment={environment} statuses={failed} isPending={false} showDetails />, options);
        expect(screen.getByText('Leader')).toBeInTheDocument();
        expect(screen.getByText('Unplugged')).toBeInTheDocument();
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
