import { ActionButton, AlertDialog, DialogTrigger } from '@geti/ui';
import { useLocalStorage } from 'usehooks-ts';

import { $api } from '../../api/client';

import classes from './robot-form/form.module.scss';

export const IdentifyRobotButton = ({ port_id }: { port_id: string }) => {
    const [confirmIdentify, setConfirmIdentify] = useLocalStorage('geti_action_confirm_identify', true);
    const { data: robots } = $api.useSuspenseQuery('get', '/api/hardware/robots');
    const identifyMutation = $api.useMutation('post', '/api/hardware/identify');
    const isDisabled = identifyMutation.isPending || port_id === null;

    if (confirmIdentify === false) {
        return (
            <ActionButton
                isDisabled={isDisabled}
                UNSAFE_className={classes.actionButton}
                onPress={() => {
                    const body = robots.find((m) => m.serial_id === port_id);

                    if (isDisabled || body === undefined) {
                        return;
                    }

                    identifyMutation.mutate({ body });
                }}
            >
                Identify
            </ActionButton>
        );
    }

    return (
        <DialogTrigger type='modal'>
            <ActionButton isDisabled={isDisabled} UNSAFE_className={classes.actionButton} onPress={() => {}}>
                Identify
            </ActionButton>

            {() => {
                return (
                    <AlertDialog
                        title='Identify robot'
                        variant='warning'
                        primaryActionLabel='Confirm'
                        onPrimaryAction={() => {
                            const body = robots.find((m) => m.serial_id === port_id);

                            if (isDisabled || body === undefined) {
                                return;
                            }

                            identifyMutation.mutate({ body });
                        }}
                    >
                        To identify your robot we will connect to it and move its gripper joint, is this ok?
                    </AlertDialog>
                );
            }}
        </DialogTrigger>
    );
};
