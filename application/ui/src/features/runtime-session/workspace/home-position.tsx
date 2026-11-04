import { useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    Button,
    ButtonGroup,
    Checkbox,
    Content,
    Dialog,
    DialogContainer,
    Flex,
    Heading,
    Item,
    Menu,
    MenuTrigger,
    Text,
} from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { useRuntimeSession } from '../runtime-session-provider';
import { useHomePosition } from './home-position-provider';
import { useSessionMode } from './session-mode';

import classes from './home-position.module.css';

const ConfirmHomeReturn = ({
    confirm,
    close,
    isPending,
}: {
    confirm: () => void;
    close: () => void;
    isPending: boolean;
}) => {
    const [isClear, setIsClear] = useState(false);
    return (
        <Dialog>
            <Heading>Return the follower to home?</Heading>
            <Content>
                <Flex direction='column' gap='size-150'>
                    <Text>
                        This stops model or teleoperation control and moves the follower slowly to your saved pose. The
                        gripper stays at its current position.
                    </Text>
                    <Text>
                        Secure anything the gripper is holding; preserving position does not guarantee grip force.
                    </Text>
                    <Text>
                        The return path is not collision-checked. Keep people, objects, and cables clear, and be ready
                        to stop the robot.
                    </Text>
                    <Checkbox isSelected={isClear} onChange={setIsClear}>
                        I have checked that the return path is clear
                    </Checkbox>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button variant='accent' isDisabled={!isClear || isPending} onPress={confirm}>
                    Return to home
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

/** Home is an explicit operator-marked pose, not a third operating mode or a collision guarantee. */
export const HomePosition = () => {
    const { state } = useRuntimeSession();
    const {
        pose,
        invalid,
        warning,
        capability,
        feedback: motion,
        error,
        phase,
        isBusy,
        canCapture,
        canReturn,
        isRecording,
        capture,
        start,
        clear,
    } = useHomePosition();
    const { isPending, activateRecording } = useSessionMode();
    const [dialog, setDialog] = useState<'return' | 'replace' | null>(null);
    const isProblem = invalid || error != null || motion?.state === 'failed';
    const captured = pose ? new Date(pose.captured_at).toLocaleString() : null;
    const confirmReturn = () => {
        if (!canReturn || isPending) return;
        setDialog(null);
        void start().then((accepted) => {
            if (accepted) activateRecording();
        });
    };

    return (
        <div className={classes.home} data-problem={isProblem || undefined} role='region' aria-label='Home position'>
            <div className={classes.header}>
                <Text UNSAFE_className={classes.title}>Home position</Text>
                {(pose != null || invalid) && (
                    <MenuTrigger>
                        <ActionButton
                            isQuiet
                            aria-label='Home position actions'
                            isDisabled={isBusy || isPending || isRecording}
                        >
                            <MoreMenu />
                        </ActionButton>
                        <Menu
                            disabledKeys={!canCapture ? ['replace'] : []}
                            onAction={(key) => (key === 'replace' ? setDialog('replace') : clear())}
                        >
                            <Item key='replace'>Replace with current pose…</Item>
                            <Item key='clear'>Clear saved home</Item>
                        </Menu>
                    </MenuTrigger>
                )}
            </div>
            {capability == null ? (
                <Text UNSAFE_className={classes.hint}>
                    Available for calibrated, serial-identified SO-101 followers with supported joint units.
                </Text>
            ) : (
                <>
                    <Text UNSAFE_className={classes.hint}>
                        {pose
                            ? `${warning ? 'Home set for this session' : 'Saved on this browser'} · ${captured}`
                            : 'Position the follower, pause teleoperation, then save its current pose.'}
                    </Text>
                    {invalid && (
                        <Text UNSAFE_className={classes.error}>
                            Saved home is invalid or does not match this robot or calibration. Set a new home.
                        </Text>
                    )}
                    {pose ? (
                        <>
                            <Text UNSAFE_className={classes.hint}>
                                User-marked pose. Check the path before every return.
                            </Text>
                            <Button
                                variant='secondary'
                                isDisabled={!canReturn || isPending}
                                onPress={() => setDialog('return')}
                            >
                                Return to home
                            </Button>
                        </>
                    ) : (
                        <Button
                            variant='secondary'
                            isPending={phase === 'capturing'}
                            isDisabled={!canCapture || isPending}
                            onPress={() => void capture()}
                        >
                            Set current pose as home
                        </Button>
                    )}
                    {!pose && state?.control != null && (
                        <Text UNSAFE_className={classes.hint}>
                            Pause model or teleoperation control before setting home.
                        </Text>
                    )}
                    {isRecording && (
                        <Text UNSAFE_className={classes.hint}>
                            Save or discard the episode before using home position.
                        </Text>
                    )}
                    {motion?.state !== 'moving' && motion?.message && (
                        <Text UNSAFE_className={motion.state === 'failed' ? classes.error : classes.hint}>
                            {motion.message}
                        </Text>
                    )}
                </>
            )}
            {warning && <Text UNSAFE_className={classes.hint}>{warning}</Text>}
            {error && <Text UNSAFE_className={classes.error}>{error}</Text>}
            <DialogContainer onDismiss={() => setDialog(null)}>
                {dialog === 'return' && (
                    <ConfirmHomeReturn
                        close={() => setDialog(null)}
                        confirm={confirmReturn}
                        isPending={isPending || !canReturn}
                    />
                )}
                {dialog === 'replace' && (
                    <AlertDialog
                        title='Replace the saved home position?'
                        variant='warning'
                        primaryActionLabel='Replace home'
                        secondaryActionLabel='Cancel'
                        onPrimaryAction={() => {
                            setDialog(null);
                            if (canCapture) void capture();
                        }}
                        onSecondaryAction={() => setDialog(null)}
                    >
                        The current follower pose will replace the saved home. No return motion will start.
                    </AlertDialog>
                )}
            </DialogContainer>
        </div>
    );
};

/** Keep Stop in the live workspace even when the Environment card has scrolled out of view. */
export const HomeReturnBanner = () => {
    const { isReturning, phase, motion, cancel } = useHomePosition();
    if (!isReturning) return null;
    const progress = phase === 'starting' ? 0 : Math.round((motion?.progress ?? 0) * 100);
    return (
        <div className={classes.banner} role='region' aria-label='Home return controls'>
            <div className={classes.bannerCopy}>
                <span className={classes.title} role='status'>
                    {phase === 'stopping'
                        ? 'Stopping home return…'
                        : phase === 'starting'
                          ? 'Starting home return…'
                          : 'Returning to home…'}
                </span>
                <Text UNSAFE_className={classes.hint}>Follower motion only · gripper position preserved</Text>
                <div
                    className={classes.progress}
                    role='progressbar'
                    aria-label='Home return progress'
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={progress}
                >
                    <span style={{ width: `${progress}%` }} />
                </div>
            </div>
            <Button
                variant='negative'
                isPending={phase === 'stopping'}
                isDisabled={phase === 'stopping'}
                onPress={() => void cancel()}
            >
                Stop return
            </Button>
        </div>
    );
};
