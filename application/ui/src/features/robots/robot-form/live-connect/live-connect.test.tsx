import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaHttpValidationError } from '../../../../api/openapi-spec';
import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { getMockedTrossenRobot } from '../../../../test-utils/mocks/mock-robot';
import { getMockedRobotCatalogDefinition } from '../../../../test-utils/mocks/mock-robot-catalog-definition';
import { render } from '../../../../test-utils/render';
import { RobotModelsProvider } from '../../robot-models-context';
import { CreateRobotForm } from '../create-form';
import { Preview } from '../preview';
import { RobotFormProvider } from '../provider';
import { UpdateRobotForm } from '../update-form';
import { LiveConnectProvider } from './provider';

const sendJsonMessage = rs.fn();
let options: { onMessage?: (event: MessageEvent) => void; onClose?: () => void } = {};
rs.mock('react-use-websocket', () => ({
    default: (_url: string, nextOptions: typeof options) => {
        options = nextOptions;
        return { sendJsonMessage, readyState: 1 };
    },
    ReadyState: { OPEN: 1 },
}));

interface Command {
    command: string;
    request_id: string;
    robot?: { payload: Record<string, unknown> };
}
const commands = () => sendJsonMessage.mock.calls.map(([command]) => command as Command);
const latest = (command: string) =>
    commands()
        .filter((message) => message.command === command)
        .at(-1);
const deliver = (message: object) => act(() => options.onMessage?.({ data: JSON.stringify(message) } as MessageEvent));

const ROBOT_TYPE = 'Trossen_WidowXAI_Follower';
const PROJECT_ID = 'project-1';
const setup = (update = false) => {
    let saves = 0;
    server.use(
        http.get('/api/robots/catalog', () => HttpResponse.json([getMockedRobotCatalogDefinition()])),
        http.get('/api/robots/catalog/{robot_type}/schema', () =>
            HttpResponse.json({
                type: 'object',
                properties: { connection_string: { type: 'string', title: 'Robot IP address' } },
                required: ['connection_string'],
            })
        ),
        http.post('/api/robots/catalog/{robot_type}/validate', async ({ request }) =>
            HttpResponse.json(await request.json())
        ),
        http.post('/api/projects/{project_id}/robots', async ({ request }) => {
            saves += 1;
            const body = await request.json();
            if (body.type !== ROBOT_TYPE) throw new Error('Unexpected test robot type');
            return HttpResponse.json<ReturnType<typeof getMockedTrossenRobot>>(
                getMockedTrossenRobot({ id: body.id, name: body.name, payload: body.payload }),
                { status: 201 }
            );
        }),
        http.put('/api/projects/{project_id}/robots/{robot_id}', async ({ request }) => {
            saves += 1;
            const body = await request.json();
            if (body.type !== ROBOT_TYPE) throw new Error('Unexpected test robot type');
            return HttpResponse.json<ReturnType<typeof getMockedTrossenRobot>>(
                getMockedTrossenRobot({ id: body.id, name: body.name, payload: body.payload })
            );
        })
    );
    const view = render(
        <RobotModelsProvider>
            <RobotFormProvider robot={{ type: ROBOT_TYPE, name: 'Arm', payload: { connection_string: '192.0.2.1' } }}>
                <LiveConnectProvider>
                    {update ? <UpdateRobotForm /> : <CreateRobotForm />}
                    <Preview />
                </LiveConnectProvider>
            </RobotFormProvider>
        </RobotModelsProvider>,
        {
            route: `/projects/${PROJECT_ID}/robots/${update ? 'robot-1/edit' : 'new'}`,
            path: update ? '/projects/:project_id/robots/:robot_id/*' : '/projects/:project_id/robots/*',
        }
    );
    return { ...view, user: userEvent.setup(), saves: () => saves };
};

const enable = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(await screen.findByRole('button', { name: 'Enable Live connect' }));
    await user.click(within(await screen.findByRole('alertdialog')).getByRole('button', { name: 'Enable' }));
    await waitFor(() => expect(latest('connect')).toBeDefined());
};

beforeEach(() => {
    sendJsonMessage.mockClear();
    options = {};
});

describe('Live robot setup', () => {
    it('requires consent and fresh observations, then awaits release before saving', async () => {
        const { user, saves } = setup();
        expect(await screen.findByRole('button', { name: 'Add robot' })).toBeDisabled();
        expect(commands()).toEqual([]);
        await enable(user);
        expect(screen.getByRole('button', { name: 'Add robot' })).toBeDisabled();
        deliver({ event: 'observation', request_id: latest('connect')?.request_id, data: { 'shoulder.pos': 15 } });
        expect(await screen.findByText(/shoulder: 15.00 raw/)).toBeInTheDocument();
        expect(screen.getByText('No 3D model supplied by this plugin.')).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Add robot' }));
        await waitFor(() => expect(latest('disconnect')).toBeDefined());
        expect(saves()).toBe(0);
        deliver({ event: 'disconnected', request_id: latest('disconnect')?.request_id });
        await waitFor(() => expect(saves()).toBe(1));
    });

    it('invalidates verification on edits and ignores observations for the old payload', async () => {
        const { user } = setup();
        await enable(user);
        const previous = latest('connect');
        deliver({ event: 'observation', request_id: previous?.request_id, data: { 'shoulder.pos': 15 } });
        await user.clear(screen.getByRole('textbox', { name: /Robot IP address/ }));
        await user.type(screen.getByRole('textbox', { name: /Robot IP address/ }), '192.0.2.2');
        expect(screen.getByRole('button', { name: 'Add robot' })).toBeDisabled();
        await waitFor(() => expect(latest('disconnect')).toBeDefined());
        deliver({ event: 'observation', request_id: previous?.request_id, data: { 'shoulder.pos': 16 } });
        expect(screen.getByRole('button', { name: 'Add robot' })).toBeDisabled();
        expect(commands().filter(({ command }) => command === 'connect')).toHaveLength(1);
        deliver({ event: 'disconnected', request_id: latest('disconnect')?.request_id });
        await waitFor(() => expect(commands().filter(({ command }) => command === 'connect')).toHaveLength(2));
        expect(latest('connect')?.robot?.payload.connection_string).toBe('192.0.2.2');
    });

    it('allows confirmed offline saving without hardware commands', async () => {
        const { user, saves } = setup();
        const offline = await screen.findByRole('button', { name: 'Save without verification' });
        await waitFor(() => expect(offline).toBeEnabled());
        await user.click(offline);
        expect(saves()).toBe(0);
        await user.click(
            within(await screen.findByRole('alertdialog')).getByRole('button', { name: 'Save without verification' })
        );
        await waitFor(() => expect(saves()).toBe(1));
        expect(commands()).toEqual([]);
    });

    it('reports a connection error without automatically retrying the same payload', async () => {
        const { user } = setup();
        await enable(user);
        deliver({
            event: 'error',
            request_id: latest('connect')?.request_id,
            message: 'Device is already in use',
            error_code: 'robot_device_already_owned',
        });
        expect(await screen.findByRole('alert')).toHaveTextContent('Device is already in use');
        expect(screen.getByRole('button', { name: 'Add robot' })).toBeDisabled();
        expect(commands().filter(({ command }) => command === 'connect')).toHaveLength(1);
    });

    it('does not reconnect when only the display name changes', async () => {
        const { user } = setup();
        await enable(user);
        deliver({ event: 'observation', request_id: latest('connect')?.request_id, data: { 'shoulder.pos': 15 } });
        await user.type(screen.getByRole('textbox', { name: /Robot name/ }), ' renamed');
        expect(commands().filter(({ command }) => command === 'connect')).toHaveLength(1);
        expect(latest('disconnect')).toBeUndefined();
        expect(screen.getByRole('button', { name: 'Add robot' })).toBeEnabled();
    });

    it('gates updates on a healthy live connection too', async () => {
        const { user } = setup(true);
        expect(await screen.findByRole('button', { name: 'Update robot' })).toBeDisabled();
        await enable(user);
        deliver({ event: 'observation', request_id: latest('connect')?.request_id, data: { 'shoulder.pos': 15 } });
        expect(screen.getByRole('button', { name: 'Update robot' })).toBeEnabled();
    });

    it('retains form data and displays persistence errors', async () => {
        const { user } = setup();
        server.use(
            http.post('/api/projects/{project_id}/robots', () =>
                HttpResponse.json<SchemaHttpValidationError>(
                    { detail: [{ loc: ['body', 'name'], msg: 'Could not save the robot', type: 'value_error' }] },
                    { status: 400 }
                )
            )
        );
        const offline = await screen.findByRole('button', { name: 'Save without verification' });
        await waitFor(() => expect(offline).toBeEnabled());
        await user.click(offline);
        await user.click(
            within(await screen.findByRole('alertdialog')).getByRole('button', { name: 'Save without verification' })
        );
        expect(await screen.findByRole('alert')).toHaveTextContent('Could not save the robot');
        expect(screen.getByRole('textbox', { name: /Robot name/ })).toHaveValue('Arm');
        expect(commands()).toEqual([]);
    });
});
