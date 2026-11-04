import { screen, within } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { SchemaSessionState } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedFollowerRobot, getMockedLeaderRobot } from '../../../test-utils/mocks/mock-robot';
import { getMockedSessionEnvironment } from '../../../test-utils/mocks/mock-runtime-session';
import { render } from '../../../test-utils/render';
import { TELEOP_CONTROL } from '../cells/robot-cell';
import { Statuses } from '../runtime-session-provider';
import { EnvironmentRobots } from './environment-robots';

const statuses: Statuses = {
    'environment/follower': { type: 'status', component: 'environment/follower', state: 'running' },
    'environment/leader': { type: 'status', component: 'environment/leader', state: 'running' },
};
const renderRobots = (control: SchemaSessionState['control'] = null, overrides: Statuses = {}) =>
    render(
        <EnvironmentRobots
            environment={getMockedSessionEnvironment({ leader_id: 'leader-1' })}
            statuses={{ ...statuses, ...overrides }}
            control={control}
        />,
        { route: '/projects/project-1', path: '/projects/:project_id' }
    );

beforeEach(() =>
    server.use(
        http.get('/api/projects/{project_id}/robots/{robot_id}', ({ params }) =>
            HttpResponse.json(params.robot_id === 'leader-1' ? getMockedLeaderRobot() : getMockedFollowerRobot())
        )
    )
);

it('shows robot identities and quiet readiness when control is held', async () => {
    renderRobots();
    expect(await screen.findByText('Workbench follower')).toBeInTheDocument();
    expect(await screen.findByText('Workbench leader')).toBeInTheDocument();
    expect(screen.getAllByText('Ready')).toHaveLength(2);
    expect(screen.queryByText('Active')).not.toBeInTheDocument();
});

it('marks both robots active during teleoperation', () => {
    renderRobots(TELEOP_CONTROL);
    expect(within(screen.getByRole('group', { name: 'Follower robot' })).getByText('Active')).toBeInTheDocument();
    expect(within(screen.getByRole('group', { name: 'Leader robot' })).getByText('Active')).toBeInTheDocument();
});

it('marks only the follower active during model control', () => {
    renderRobots({
        type: 'model',
        task: 'Pick',
        hz: 30,
        execution: { mode: 'sync', request_threshold: 0.5 },
        grace_s: 3,
    });
    expect(within(screen.getByRole('group', { name: 'Follower robot' })).getByText('Active')).toBeInTheDocument();
    expect(within(screen.getByRole('group', { name: 'Leader robot' })).getByText('Ready')).toBeInTheDocument();
});

it('shows disconnected devices and never marks them active', () => {
    renderRobots(TELEOP_CONTROL, {
        'environment/leader': {
            type: 'status',
            component: 'environment/leader',
            state: 'failed',
            error: 'Check the USB cable',
        },
    });
    const leader = screen.getByRole('group', { name: 'Leader robot' });
    expect(within(leader).getByText('Disconnected')).toBeInTheDocument();
    expect(within(leader).getByText('Check the USB cable')).toBeInTheDocument();
    expect(within(leader).queryByText('Active')).not.toBeInTheDocument();
});

it('does not imply activity when control has failed', () => {
    renderRobots(TELEOP_CONTROL, {
        control: { type: 'status', component: 'control', state: 'failed', error: 'Control stopped' },
    });
    expect(screen.queryByText('Active')).not.toBeInTheDocument();
});
