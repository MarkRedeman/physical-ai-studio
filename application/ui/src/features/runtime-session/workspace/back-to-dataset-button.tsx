import { useRef, useState } from 'react';

import { ActionButton, ProgressCircle, toast, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';

import { paths } from '../../../paths';
import { useProjectId } from '../../projects/use-project';
import { ERROR_TOAST, useRuntimeSession } from '../runtime-session-provider';
import { useHomePositionBusy } from './home-position-provider';

/** Finish copying session episodes into the selected dataset before navigating away. */
export const BackToDatasetButton = ({ dataset_id }: { dataset_id: string }) => {
    const { project_id } = useProjectId();
    const { state, send } = useRuntimeSession();
    const queryClient = useQueryClient();
    const navigate = useNavigate();
    const [isFinishing, setIsFinishing] = useState(false);
    const inFlight = useRef(false);
    const isRecording = state?.dataset?.is_recording ?? false;
    const homeBusy = useHomePositionBusy();

    const finish = async () => {
        if (inFlight.current || isRecording || homeBusy) return;
        inFlight.current = true;
        setIsFinishing(true);
        const selectedDatasetId = state?.dataset?.dataset_id ?? dataset_id;
        try {
            if (state?.dataset != null) {
                const ack = await send({ command: 'unload_dataset' });
                if (!ack.ok) {
                    toast.negative(
                        ack.error?.message ?? 'Could not finish saving the recording. Please try again.',
                        ERROR_TOAST
                    );
                    return;
                }
            }
            await queryClient.invalidateQueries({ queryKey: ['get', '/api/dataset/{dataset_id}/episodes'] });
            navigate(paths.project.datasets.show({ project_id, dataset_id: selectedDatasetId }));
        } finally {
            inFlight.current = false;
            setIsFinishing(false);
        }
    };

    return (
        <TooltipTrigger>
            <ActionButton
                isQuiet
                aria-label='Back to dataset'
                isDisabled={isFinishing || isRecording || homeBusy}
                onPress={() => void finish()}
            >
                {isFinishing ? (
                    <ProgressCircle size='S' isIndeterminate aria-label='Saving the recording' />
                ) : (
                    <ChevronLeft />
                )}
            </ActionButton>
            <Tooltip>{isRecording ? 'Save or discard this episode before leaving' : 'Back to dataset'}</Tooltip>
        </TooltipTrigger>
    );
};
