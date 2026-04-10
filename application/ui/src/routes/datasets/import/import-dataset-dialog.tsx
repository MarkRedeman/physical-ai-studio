import { Suspense, useEffect, useState } from 'react';

import { Content, Dialog, Divider, Heading, Loading } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';

import { SchemaDatasetImportJob } from '../../../api/openapi-spec';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { ImportStepDetectionFailed } from './import-step-detection-failed';
import { ImportStepEditing } from './import-step-editing';
import { ImportStepInProgress } from './import-step-in-progress';
import { ImportStepReadyToFinalize } from './import-step-ready-to-finalize';
import { useDatasetImportJobQuery, type FinalizeFields } from './use-dataset-import-job-state';

const DETECTION_STEPS = ['queued_for_detection', 'detecting_format', 'building_manifest_draft'];

interface InternalImportDatasetDialogProps {
    project_id: string;
    onClose: () => void;
    initialJobId?: string;
    onPendingJobDismissed?: (jobId: string) => void;
    onImportCompleted?: (datasetId: string) => void;
}

interface ImportDatasetDialogProps {
    onClose: () => void;
    initialJobId?: string;
    onPendingJobDismissed?: (jobId: string) => void;
    onImportCompleted?: (datasetId: string) => void;
}

const DatasetImportHeading = ({ importJob }: { importJob: SchemaDatasetImportJob | undefined }) => {
    const importPayload = importJob?.payload;

    const payloadStep = importPayload?.step;
    const formatHint = importPayload?.format_hint;

    const isDetectionFailed =
        importJob?.status === 'failed' && payloadStep !== undefined && DETECTION_STEPS.includes(String(payloadStep));

    if (isDetectionFailed === false) {
        return <Heading>Import dataset</Heading>;
    }

    const usedAutoDetection = formatHint === 'auto' || formatHint === undefined;

    return (
        <Heading>
            {usedAutoDetection ? 'Automatic format detection failed' : 'Selected format validation failed'}
        </Heading>
    );
};

const getStatus = (importJob: SchemaDatasetImportJob | undefined) => {
    const importPayload = importJob?.payload;
    const payloadStep = importPayload?.step;

    const isEditing = importJob?.id === undefined || payloadStep === 'awaiting_archive_upload';
    const isDetectionFailed =
        importJob?.status === 'failed' && payloadStep !== undefined && DETECTION_STEPS.includes(String(payloadStep));
    const isAwaitingDetection = payloadStep !== undefined && DETECTION_STEPS.includes(String(payloadStep));
    const isReadyToFinalize = payloadStep === 'awaiting_user_review';
    const isAwaitingImport =
        payloadStep === 'queued_for_import' || payloadStep === 'importing_dataset' || payloadStep === 'completed';

    return {
        isEditing,
        isDetectionFailed,
        isAwaitingDetection,
        isReadyToFinalize,
        isAwaitingImport,
    };
};

const useOnImportDone = (importJob: SchemaDatasetImportJob | undefined, onImportDone: (datasetId: string) => void) => {
    const completedDatasetId = importJob?.status === 'completed' ? importJob.payload?.result_dataset_id : undefined;
    useEffect(() => {
        if (completedDatasetId) {
            onImportDone(completedDatasetId);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [completedDatasetId]);
};

const InternalImportDatasetDialog = ({
    project_id,
    onClose,
    initialJobId,
    onPendingJobDismissed,
    onImportCompleted,
}: InternalImportDatasetDialogProps) => {
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const [datasetName, setDatasetName] = useState('');
    const [finalizeFields, setFinalizeFields] = useState<FinalizeFields>({
        defaultTask: '',
        environmentId: undefined,
    });
    const [importJobId, setImportJobId] = useState<string | undefined>(initialJobId);

    const importJobQuery = useDatasetImportJobQuery(importJobId);
    const importJob = importJobQuery.data as SchemaDatasetImportJob | undefined;
    const { isAwaitingDetection, isAwaitingImport, isDetectionFailed, isEditing, isReadyToFinalize } =
        getStatus(importJob);

    const onDialogClose = () => {
        if (importJobId && isReadyToFinalize) {
            onPendingJobDismissed?.(importJobId);
        }
        onClose();
    };

    useOnImportDone(importJob, (datasetId: string) => {
        void queryClient.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}'] });
        void queryClient.invalidateQueries({ queryKey: ['get', '/api/jobs'] });
        onImportCompleted?.(datasetId);
        onDialogClose();
        navigate(paths.project.datasets.show({ project_id, dataset_id: datasetId }));
    });

    return (
        <Dialog>
            <DatasetImportHeading importJob={importJob} />
            <Divider />

            {isEditing && (
                <ImportStepEditing
                    importJob={importJob}
                    project_id={project_id}
                    onClose={onDialogClose}
                    datasetName={datasetName}
                    onDatasetNameChange={setDatasetName}
                    onUploaded={setImportJobId}
                />
            )}

            {importJob !== undefined && isAwaitingDetection && (
                <ImportStepInProgress statusMessage='Upload accepted. Waiting for server-side dataset detection...' />
            )}

            {importJob !== undefined && isAwaitingImport && (
                <ImportStepInProgress statusMessage={importJob?.message ?? 'Importing dataset...'} />
            )}

            {importJob !== undefined && isDetectionFailed && (
                <ImportStepDetectionFailed importJob={importJob} onClose={onDialogClose} />
            )}

            {importJob !== undefined && isReadyToFinalize && importJobId && (
                // TODO: Rename ImportStepUserReview
                <ImportStepReadyToFinalize
                    importJob={importJob}
                    project_id={project_id}
                    onClose={onClose}
                    fields={finalizeFields}
                    onFieldsChange={setFinalizeFields}
                />
            )}
        </Dialog>
    );
};

export const ImportDatasetDialog = ({
    onClose,
    initialJobId,
    onPendingJobDismissed,
    onImportCompleted,
}: ImportDatasetDialogProps) => {
    const { project_id } = useProjectId();

    return (
        <Suspense
            fallback={
                <Dialog>
                    <Heading>Import dataset</Heading>
                    <Divider />
                    <Content>
                        <Loading />
                    </Content>
                </Dialog>
            }
        >
            <InternalImportDatasetDialog
                project_id={project_id}
                onClose={onClose}
                initialJobId={initialJobId}
                onPendingJobDismissed={onPendingJobDismissed}
                onImportCompleted={onImportCompleted}
            />
        </Suspense>
    );
};

export type { ImportDatasetDialogProps };
