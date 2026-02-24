import { useCallback, useState } from 'react';

import { Button, FileTrigger, Flex, Heading, Loading, Text } from '@geti/ui';

import { InlineAlert } from '../shared/inline-alert';
import { StatusBadge } from '../shared/status-badge';
import type { CalibrationResult } from './use-setup-websocket';
import { CalibrationPhase, useSetupActions, useSetupState, WizardStep } from './wizard-provider';

import classes from '../shared/setup-wizard.module.scss';

const MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];

/**
 * Calibration step:
 * 1. Ask user to center the robot, then apply homing offsets
 * 2. Ask user to move all joints through range while showing live position table
 * 3. Finalize and write calibration to motor EEPROM
 */
export const CalibrationStep = () => {
    const { wsState, calibrationPhase: phase } = useSetupState();
    const { goNext, goBack, markCompleted, commands, setCalibrationPhase } = useSetupActions();

    // Tracks whether we've already auto-started recording after homing completes.
    // Reset when going back to instructions for recalibration.
    const [prevHomingDone, setPrevHomingDone] = useState(false);

    // Tracks whether we've already sent the enter_calibration command.
    // The backend streams raw positions in CALIBRATION_INSTRUCTIONS phase.
    const [enteredCalibration, setEnteredCalibration] = useState(false);

    // Helper that updates phase in context.
    // Replaces the old local state + onPhaseChange callback.
    const changePhase = useCallback(
        (newPhase: CalibrationPhase) => {
            setCalibrationPhase(newPhase);
            // Reset homing tracking when going back to instructions,
            // so auto-start recording can fire again on recalibration.
            if (newPhase === 'instructions') {
                setPrevHomingDone(false);
                setEnteredCalibration(false);
            }
        },
        [setCalibrationPhase]
    );

    // Tell the backend to enter CALIBRATION_INSTRUCTIONS phase so it starts
    // streaming raw positions.  Uses the "adjusting state during render" pattern.
    if (phase === 'instructions' && !enteredCalibration) {
        setEnteredCalibration(true);
        commands.enterCalibration();
    }

    const { homingResult, positions, calibrationResult, error } = wsState;

    const handleApplyHoming = () => {
        changePhase('homing');
        commands.startHoming();
    };

    // After homing result arrives, move to recording phase
    const homingDone = homingResult !== null;
    const isRecording = phase === 'recording';
    const calibrationDone = calibrationResult !== null;

    // Auto-start recording as soon as homing offsets are applied.
    // Uses the "adjusting state during render" pattern instead of useEffect.
    // See https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes
    if (phase === 'homing' && homingDone && !prevHomingDone) {
        setPrevHomingDone(true);
        changePhase('recording');
        commands.startRecording();
    }

    const handleStopRecording = () => {
        commands.stopRecording();
        changePhase('done');
    };

    // Upload calibration file handling
    const [uploadError, setUploadError] = useState<string | null>(null);

    const handleCalibrationUpload = useCallback(
        (files: FileList | null) => {
            const file = files?.[0];
            if (!file) return;

            setUploadError(null);

            const reader = new FileReader();
            reader.onload = () => {
                try {
                    const parsed: unknown = JSON.parse(reader.result as string);
                    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
                        setUploadError('File must contain a JSON object with motor calibration data.');
                        return;
                    }

                    const calibration = parsed as Record<string, Record<string, unknown>>;
                    const requiredFields = ['id', 'drive_mode', 'homing_offset', 'range_min', 'range_max'] as const;

                    // Validate each motor entry has all required numeric fields
                    for (const [motorName, values] of Object.entries(calibration)) {
                        if (typeof values !== 'object' || values === null || Array.isArray(values)) {
                            setUploadError(`Motor "${motorName}" must be an object with calibration fields.`);
                            return;
                        }
                        for (const field of requiredFields) {
                            if (typeof values[field] !== 'number') {
                                setUploadError(
                                    `Motor "${motorName}" is missing or has invalid "${field}" (expected a number).`
                                );
                                return;
                            }
                        }
                    }

                    // Validate motor names match expected set
                    const provided = Object.keys(calibration).sort();
                    const expected = [...MOTOR_NAMES].sort();
                    if (provided.join(',') !== expected.join(',')) {
                        const missing = expected.filter((m) => !provided.includes(m));
                        const extra = provided.filter((m) => !expected.includes(m));
                        const parts: string[] = [];
                        if (missing.length) parts.push(`missing: ${missing.join(', ')}`);
                        if (extra.length) parts.push(`unexpected: ${extra.join(', ')}`);
                        setUploadError(`Motor name mismatch — ${parts.join('; ')}.`);
                        return;
                    }

                    commands.applyCalibration(calibration as CalibrationResult['calibration']);
                    changePhase('done');
                } catch {
                    setUploadError('Failed to parse file as JSON. Please select a valid calibration JSON file.');
                }
            };
            reader.onerror = () => {
                setUploadError('Failed to read the selected file.');
            };
            reader.readAsText(file);
        },
        [commands, changePhase]
    );

    if (error && phase !== 'done') {
        return (
            <Flex direction='column' gap='size-200'>
                <InlineAlert variant='error'>
                    <strong>Error:</strong> {error}
                </InlineAlert>
                <Flex justifyContent='end'>
                    <Button variant='secondary' onPress={() => changePhase('instructions')}>
                        Try Again
                    </Button>
                </Flex>
            </Flex>
        );
    }

    return (
        <Flex direction='column' gap='size-300'>
            {/* Phase 1: Instructions — center the robot */}
            {phase === 'instructions' && (
                <>
                    {/* Live joint positions while centering */}
                    <div className={classes.sectionCard}>
                        <Flex direction='column' gap='size-100'>
                            <Heading level={4}>Joint Positions</Heading>
                            <table className={classes.rangeTable}>
                                <thead>
                                    <tr>
                                        <th>Joint</th>
                                        <th>Position</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {MOTOR_NAMES.map((name) => {
                                        const motor = positions?.motors[name];
                                        return (
                                            <tr key={name}>
                                                <td>{name}</td>
                                                <td>{motor?.position ?? '-'}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </Flex>
                    </div>

                    <InlineAlert variant='info'>
                        Move the robot arm to the <strong>center of its range of motion</strong> for all joints. Each
                        joint should be roughly in the middle position. When ready, click &ldquo;Apply Homing
                        Offsets&rdquo;.
                    </InlineAlert>
                    <InlineAlert variant='warning'>
                        Make sure the robot is <strong>powered on</strong> but <strong>not under torque</strong> — you
                        should be able to move the joints freely by hand while the live positions update above.
                    </InlineAlert>

                    {uploadError && <InlineAlert variant='error'>{uploadError}</InlineAlert>}

                    <Flex justifyContent='space-between'>
                        <Button variant='secondary' onPress={goBack}>
                            Back
                        </Button>
                        <Flex gap='size-100'>
                            <FileTrigger acceptedFileTypes={['.json']} onSelect={handleCalibrationUpload}>
                                <Button variant='secondary'>Upload Calibration File</Button>
                            </FileTrigger>
                            <Button variant='accent' onPress={handleApplyHoming}>
                                Apply Homing Offsets
                            </Button>
                        </Flex>
                    </Flex>
                </>
            )}

            {/* Phase 2: Homing — waiting for result, then auto-starts recording */}
            {phase === 'homing' && (
                <Flex
                    direction='column'
                    gap='size-300'
                    alignItems='center'
                    justifyContent='center'
                    minHeight='size-3000'
                >
                    <Loading mode='inline' />
                    <Heading level={4} margin={0}>
                        Applying Homing Offsets
                    </Heading>
                    <Text
                        UNSAFE_style={{
                            color: 'var(--spectrum-global-color-gray-600)',
                            textAlign: 'center',
                            maxWidth: 400,
                        }}
                    >
                        Reading current motor positions and computing center offsets. This takes a few seconds.
                    </Text>
                </Flex>
            )}

            {/* Phase 3: Recording — live position table */}
            {isRecording && (
                <>
                    <div className={classes.sectionCard}>
                        <Flex direction='column' gap='size-100'>
                            <Flex justifyContent='space-between' alignItems='center'>
                                <Heading level={4}>Range of Motion Recording</Heading>
                                <StatusBadge variant='scanning'>Recording...</StatusBadge>
                            </Flex>
                            <table className={classes.rangeTable}>
                                <thead>
                                    <tr>
                                        <th>Joint</th>
                                        <th>Min</th>
                                        <th>Position</th>
                                        <th>Max</th>
                                        <th>Range</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {MOTOR_NAMES.map((name) => {
                                        const motor = positions?.motors[name];
                                        if (!motor) {
                                            return (
                                                <tr key={name}>
                                                    <td>{name}</td>
                                                    <td>-</td>
                                                    <td>-</td>
                                                    <td>-</td>
                                                    <td>-</td>
                                                </tr>
                                            );
                                        }
                                        const range = (motor.max ?? motor.position) - (motor.min ?? motor.position);
                                        return (
                                            <tr key={name}>
                                                <td>{name}</td>
                                                <td>{motor.min ?? '-'}</td>
                                                <td>{motor.position}</td>
                                                <td>{motor.max ?? '-'}</td>
                                                <td>{range > 0 ? range : '-'}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </Flex>
                    </div>
                    <InlineAlert variant='info'>
                        Move <strong>every joint</strong> slowly through its complete range of motion (minimum to
                        maximum). When you have covered the full range for all joints, click &ldquo;Finish
                        Recording&rdquo;.
                    </InlineAlert>
                    <Flex justifyContent='end'>
                        <Button variant='accent' onPress={handleStopRecording}>
                            Finish Recording
                        </Button>
                    </Flex>
                </>
            )}

            {/* Phase 4: Done — calibration written */}
            {phase === 'done' && (
                <>
                    {calibrationDone ? (
                        <Flex direction='column' gap='size-200'>
                            <div className={classes.sectionCard}>
                                <Flex direction='column' gap='size-100'>
                                    <Heading level={4}>Calibration Values</Heading>
                                    <table className={classes.rangeTable}>
                                        <thead>
                                            <tr>
                                                <th>Joint</th>
                                                <th>Homing Offset</th>
                                                <th>Min</th>
                                                <th>Max</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {Object.entries(calibrationResult.calibration).map(([name, cal]) => (
                                                <tr key={name}>
                                                    <td>{name}</td>
                                                    <td>{cal.homing_offset}</td>
                                                    <td>{cal.range_min}</td>
                                                    <td>{cal.range_max}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </Flex>
                            </div>
                            <InlineAlert variant='success'>
                                Calibration has been written to the motor EEPROM and motors have been configured.
                            </InlineAlert>
                            <Flex justifyContent='space-between'>
                                <Button variant='secondary' onPress={() => changePhase('instructions')}>
                                    Back
                                </Button>
                                <Button
                                    variant='accent'
                                    onPress={() => {
                                        markCompleted(WizardStep.CALIBRATION);
                                        commands.enterVerification();
                                        goNext();
                                    }}
                                >
                                    Continue to Verification
                                </Button>
                            </Flex>
                        </Flex>
                    ) : (
                        <Flex
                            direction='column'
                            gap='size-300'
                            alignItems='center'
                            justifyContent='center'
                            minHeight='size-3000'
                        >
                            <Loading mode='inline' />
                            <Heading level={4} margin={0}>
                                Writing Calibration
                            </Heading>
                            <Text
                                UNSAFE_style={{
                                    color: 'var(--spectrum-global-color-gray-600)',
                                    textAlign: 'center',
                                    maxWidth: 400,
                                }}
                            >
                                Saving calibration data to motor EEPROM and applying PID configuration. This takes a few
                                seconds.
                            </Text>
                        </Flex>
                    )}
                </>
            )}
        </Flex>
    );
};
