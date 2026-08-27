import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { FormFields, RobotType } from './form';
import { RobotFormProvider } from './provider';

const so101Definition = {
    type: 'SO101_Follower',
    display_name: 'SO101 Follower',
    role: 'follower',
    urdf_path: '/api/robots/catalog/SO101_Follower/urdf',
    package_map: {},
    joint_map: {},
} as const;

const renderRobotTypeAndFields = () =>
    render(
        <RobotFormProvider>
            <RobotType />
            <FormFields />
        </RobotFormProvider>
    );

describe('RobotType and FormFields', () => {
    it('auto-focuses the name field on mount', async () => {
        server.use(
            http.get('/api/robots/catalog', () => HttpResponse.json([so101Definition])),
            http.get('/api/robots/catalog/{robot_type}/schema', () =>
                HttpResponse.json({ type: 'object', properties: {}, required: [] })
            )
        );

        renderRobotTypeAndFields();

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveFocus();
    });

    it('prefills the robot name with the selected type display name when the name is empty', async () => {
        server.use(
            http.get('/api/robots/catalog', () => HttpResponse.json([so101Definition])),
            http.get('/api/robots/catalog/{robot_type}/schema', () =>
                HttpResponse.json({ type: 'object', properties: {}, required: [] })
            )
        );
        const user = userEvent.setup();

        renderRobotTypeAndFields();

        await user.click(await screen.findByRole('button', { name: /Robot type/ }));
        await user.click(await screen.findByRole('option', { name: 'SO101 Follower' }));

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveValue('SO101 Follower');
    });

    it('does not overwrite an already-set robot name when changing the type', async () => {
        server.use(
            http.get('/api/robots/catalog', () => HttpResponse.json([so101Definition])),
            http.get('/api/robots/catalog/{robot_type}/schema', () =>
                HttpResponse.json({ type: 'object', properties: {}, required: [] })
            )
        );
        const user = userEvent.setup();

        render(
            <RobotFormProvider robot={{ type: 'SO101_Follower', name: 'My arm', payload: {} }}>
                <RobotType />
                <FormFields />
            </RobotFormProvider>
        );

        const nameField = await screen.findByRole('textbox', { name: /Robot name/ });
        expect(nameField).toHaveValue('My arm');

        await user.click(screen.getByRole('button', { name: /Robot type/ }));
        await user.click(await screen.findByRole('option', { name: 'SO101 Follower' }));

        expect(nameField).toHaveValue('My arm');
    });
});