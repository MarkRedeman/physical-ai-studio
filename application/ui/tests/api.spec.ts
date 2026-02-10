import { invariant } from 'outvariant';
import { v4 as uuidv4 } from 'uuid';

import { getClient } from '../src/api/client';
import { expect, test } from './fixtures';

const client = getClient({ baseUrl: 'http://localhost:7860' });

test.describe('/api/projects', () => {
    test('Happy path', async ({}) => {
        const projectsResponse = await client.GET('/api/projects');
        expect(projectsResponse.response.status).toBe(200);
        invariant(projectsResponse.data, 'Expected to recieve projects data');

        const oldLength = projectsResponse.data.length;
        expect(oldLength).toBeGreaterThanOrEqual(0);

        const newProject = {
            id: uuidv4(),
            name: 'Playwright API Test',
        };

        await test.step('Create a new project', async () => {
            // POST /api/projects
            const newProjectResponse = await client.POST('/api/projects', {
                body: {
                    id: newProject.id,
                    name: newProject.name,
                    datasets: [],
                },
            });

            expect(newProjectResponse.response.status).toBe(201);
            invariant(newProjectResponse.data, 'Expected to recieve project data');

            console.log(newProjectResponse.data);
            expect(newProjectResponse.data.id).toEqual(newProject.id);
            expect(newProjectResponse.data.name).toEqual(newProject.name);
            expect(newProjectResponse.data.datasets).toEqual([]);
        });

        await test.step('Verify new project was made', async () => {
            // GET /api/projects
            const projectsResponse = await client.GET('/api/projects');
            expect(projectsResponse.response.status).toBe(200);
            invariant(projectsResponse.data, 'Expected to recieve projects data');

            const newLength = projectsResponse.data.length;
            expect(newLength).toBeGreaterThanOrEqual(oldLength);

            // GET /api/projects/{project_id}
            const projectResponse = await client.GET('/api/projects/{project_id}', {
                params: { path: { project_id: newProject.id } },
            });
            expect(projectResponse.response.status).toBe(200);
            invariant(projectResponse.data, 'Expected to recieve project data');

            expect(projectResponse.data.id).toEqual(newProject.id);
            expect(projectResponse.data.name).toEqual(newProject.name);
        });

        await test.step('Delete new project', async () => {
            // DELETE /api/projects/{id}
            const deleteProjectResponse = await client.DELETE('/api/projects/{project_id}', {
                params: { path: { project_id: newProject.id } },
            });
            console.log(deleteProjectResponse);
            expect(deleteProjectResponse.response.status).toBe(204);
            //invariant(deleteProjectResponse.data, 'Expected to recieve project data');
        });
    });

    test('Sad path', async () => {
        await test.step('404', async () => {
            // GET /api/projects/{id}
            const projectResponse = await client.GET('/api/projects/{project_id}', {
                params: { path: { project_id: 'cdafb354-1739-4fc3-bc30-d6cd61d1c5b4' } },
            });
            expect(projectResponse.response.status).toBe(404);
            console.log(projectResponse);
        });

        await test.step('404', async () => {
            // GET /api/projects/{id}
            const projectResponse = await client.DELETE('/api/projects/{project_id}', {
                params: { path: { project_id: 'cdafb354-1739-4fc3-bc30-d6cd61d1c5b4' } },
            });
            expect(projectResponse.response.status).toBe(404);
        });

        await test.step('Bad ID', async () => {
            // GET /api/projects/{id}
            const projectResponse = await client.GET('/api/projects/{project_id}', {
                params: { path: { project_id: 'malvormed-id' } },
            });
            expect(projectResponse.response.status).toBe(400);

            invariant(projectResponse.error, 'Response should error');
            expect(projectResponse.error).toEqual({
                detail: 'Invalid project ID',
            });
        });
    });
});
