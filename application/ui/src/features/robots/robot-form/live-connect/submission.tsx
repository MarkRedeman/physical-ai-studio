import { useRef, useState } from 'react';

import { AlertDialog, Button, DialogTrigger } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaRobotInput } from '../../robot-types';
import { useLiveConnect } from './provider';

export const useRobotSubmission = (
    body: SchemaRobotInput | null,
    save: (snapshot: SchemaRobotInput) => Promise<void>,
    beginsSetup = false
) => {
    const live = useLiveConnect();
    const [pending, setPending] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const submitting = useRef(false);
    const valid = body !== null && live.validatedPayload !== null;
    const canSubmit = valid && (beginsSetup || live.verified) && !pending;

    const submit = async (offline = false) => {
        if (!valid || body === null || submitting.current || (!beginsSetup && !live.verified && !offline)) return;
        const snapshot = structuredClone({ ...body, payload: live.validatedPayload }) as SchemaRobotInput;
        submitting.current = true;
        setPending(true);
        setError(null);
        try {
            await live.disconnect();
            await save(snapshot);
        } catch (failure) {
            setError(
                getApiErrorMessage(failure) ?? (failure instanceof Error ? failure.message : 'Failed to save robot.')
            );
            live.resume();
        } finally {
            submitting.current = false;
            setPending(false);
        }
    };

    return { submit, pending, error, canSubmit, canSaveOffline: valid && !pending && !beginsSetup };
};

export const OfflineSaveButton = ({ disabled, onSave }: { disabled: boolean; onSave: () => void }) => {
    const { error, enabled } = useLiveConnect();
    return (
        <DialogTrigger>
            <Button variant='secondary' isDisabled={disabled}>
                Save without verification
            </Button>
            <AlertDialog
                title='Save without verification?'
                variant='warning'
                primaryActionLabel='Save without verification'
                secondaryActionLabel='Cancel'
                onPrimaryAction={onSave}
                isPrimaryActionDisabled={disabled}
            >
                {error
                    ? 'The robot could not be verified with this configuration.'
                    : enabled
                      ? 'There is no healthy live connection for this configuration.'
                      : 'This configuration has not been tested on the robot.'}{' '}
                Saving will not verify connectivity or calibration. You will need to check the connection before using
                the robot.
            </AlertDialog>
        </DialogTrigger>
    );
};
