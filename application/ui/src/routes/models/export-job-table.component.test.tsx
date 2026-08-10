import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { SchemaModelExportJob } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { ExportRow } from './export-job-table.component';

const completedExportJob: SchemaModelExportJob = {
    id: 'export-job-1',
    project_id: 'project-1',
    type: 'model_export',
    status: 'completed',
    progress: 100,
    message: 'Export completed',
    created_at: '2026-08-01T10:00:00Z',
    start_time: '2026-08-01T10:00:01Z',
    end_time: '2026-08-01T10:02:00Z',
    payload: { model_id: 'model-1', backends: ['torch', 'openvino'], compress: true },
};

const runningExportJob: SchemaModelExportJob = {
    ...completedExportJob,
    id: 'export-job-2',
    status: 'running',
    progress: 40,
    message: 'Exporting',
    end_time: null,
};

describe('ExportRow', () => {
    it('deletes an export job through the job options menu', async () => {
        const onDelete = vi.fn();
        server.use(
            http.delete('/api/jobs/{job_id}', () => {
                onDelete();
                return new HttpResponse(null, { status: 204 });
            })
        );

        const user = userEvent.setup();
        render(<ExportRow job={completedExportJob} modelName='Test model' onViewLogs={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: 'Export job options' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));

        await waitFor(() => expect(onDelete).toHaveBeenCalledTimes(1));
    });

    it('keeps Delete disabled for an export job that is still running', async () => {
        const user = userEvent.setup();
        render(<ExportRow job={runningExportJob} modelName='Test model' onViewLogs={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: 'Export job options' }));
        const deleteItem = await screen.findByRole('menuitem', { name: 'Delete' });

        expect(deleteItem).toHaveAttribute('aria-disabled', 'true');
    });

    it('opens logs from the job options menu', async () => {
        const onViewLogs = vi.fn();
        const user = userEvent.setup();
        render(<ExportRow job={completedExportJob} modelName='Test model' onViewLogs={onViewLogs} />);

        await user.click(screen.getByRole('button', { name: 'Export job options' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Logs' }));

        expect(onViewLogs).toHaveBeenCalledTimes(1);
    });
});
