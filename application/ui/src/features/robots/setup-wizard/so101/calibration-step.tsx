import { useCallback, useEffect, useState } from 'react';

import { Button, FileTrigger, Flex, Heading, Loading, Text } from '@geti-ui/ui';

import { InlineAlert } from '../shared/inline-alert';
import { StatusBadge } from '../shared/status-badge';
import { CalibrationPhase, useSetupActions, useSetupState, WizardStep } from './wizard-provider';
import { CalibrationUploadPayload } from './use-setup-websocket';

import classes from '../shared/setup-wizard.module.css';

const MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'] as const;
const CALIBRATION_FIELDS = ['id', 'drive_mode', 'homing_offset', 'range_min', 'range_max'] as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
    typeof value === 'object' && value !== null && !Array.isArray(value);

const toCalibrationPayload = (value: unknown): { payload: CalibrationUploadPayload | null; error: string | null } => {
    if (!isRecord(value)) {
        return { payload: null, error: 'Calibration file must contain a JSON object.' };
    }

    const payload: CalibrationUploadPayload = {};

    for (const motorName of MOTOR_NAMES) {
        const motorValue = value[motorName];
        if (!isRecord(motorValue)) {
            return { payload: null, error: `Calibration entry for '${motorName}' is missing or invalid.` };
        }

        const parsedMotor: Record<(typeof CALIBRATION_FIELDS)[number], number> = {
            id: 0,
            drive_mode: 0,
            homing_offset: 0,
            range_min: 0,
            range_max: 0,
        };

        for (const field of CALIBRATION_FIELDS) {
            const rawValue = motorValue[field];
            if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
                return {
                    payload: null,
                    error: `Calibration entry for '${motorName}' has invalid numeric field '${field}'.`,
                };
            }
            parsedMotor[field] = rawValue;
        }

        if (parsedMotor.range_min >= parsedMotor.range_max) {
            return {
                payload: null,
                error: `Calibration entry for '${motorName}' must have range_min < range_max.`,
            };
        }

        payload[motorName] = parsedMotor;
    }

    return { payload, error: null };
};

interface CalibrationJsonImportCardProps {
    onApply: (payload: CalibrationUploadPayload) => void;
}

const CalibrationJsonImportCard = ({ onApply }: CalibrationJsonImportCardProps) => {
    const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
    const [uploadedPayload, setUploadedPayload] = useState<CalibrationUploadPayload | null>(null);
    const [uploadError, setUploadError] = useState<string | null>(null);

    const handleUploadedFile = async (files: FileList | null) => {
        const file = files?.[0] ?? null;
        if (file === null) {
            return;
        }

        setUploadedFileName(file.name);
        setUploadError(null);
        setUploadedPayload(null);

        try {
            const rawText = await file.text();
            const parsed = JSON.parse(rawText) as unknown;
            const { payload, error: parseError } = toCalibrationPayload(parsed);

            if (parseError || !payload) {
                setUploadError(parseError ?? 'Invalid calibration file.');
                return;
            }

            setUploadedPayload(payload);
        } catch {
            setUploadError('Failed to parse JSON file. Please upload a valid calibration JSON.');
        }
    };

    return (
        <div className={classes.sectionCard}>
            <Flex direction='column' gap='size-150'>
                <Heading level={4}>Import Calibration JSON</Heading>
                <Text>
                    If you already have calibration values, upload them to apply directly to motors and skip manual
                    homing and recording.
                </Text>
                <Flex gap='size-100' alignItems='center'>
                    <FileTrigger acceptedFileTypes={['.json']} onSelect={handleUploadedFile}>
                        <Button variant='secondary'>Upload JSON</Button>
                    </FileTrigger>
                    <Text>{uploadedFileName ?? 'No file selected'}</Text>
                </Flex>
                {uploadError && <InlineAlert variant='error'>{uploadError}</InlineAlert>}

                {uploadedPayload && (
                    <>
                        <InlineAlert variant='success'>Calibration JSON parsed successfully.</InlineAlert>
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
                                {Object.entries(uploadedPayload).map(([name, cal]) => (
                                    <tr key={name}>
                                        <td>{name}</td>
                                        <td>{cal.homing_offset}</td>
                                        <td>{cal.range_min}</td>
                                        <td>{cal.range_max}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </>
                )}

                <Flex justifyContent='end'>
                    <Button variant='secondary' onPress={() => uploadedPayload && onApply(uploadedPayload)} isDisabled={!uploadedPayload}>
                        Apply Uploaded Calibration
                    </Button>
                </Flex>
            </Flex>
        </div>
    );
};

/**
 * Calibration step:
 * 1. Ask user to center the robot, then apply homing offsets
 * 2. Ask user to move all joints through range while showing live position table
 * 3. Finalize and write calibration to motor EEPROM
 */
export const CalibrationStep = () => {
    const { wsState, calibrationPhase: phase } = useSetupState();
    const { goNext, goBack, markCompleted, commands, setCalibrationPhase } = useSetupActions();

    const [recordingStarted, setRecordingStarted] = useState(false);

    // Helper that updates phase in context.
    // Replaces the old local state + onPhaseChange callback.
    const changePhase = useCallback(
        (newPhase: CalibrationPhase) => {
            setCalibrationPhase(newPhase);
            // Reset homing tracking when going back to instructions,
            // so auto-start recording can fire again on recalibration.
            if (newPhase === 'instructions') {
                setRecordingStarted(false);
            }
        },
        [setCalibrationPhase]
    );

    useEffect(() => {
        if (phase === 'instructions') {
            commands.enterCalibration();
        }
    }, [phase, commands]);

    const { homingResult, positions, calibrationResult, error } = wsState;
    const effectiveCalibration = calibrationResult;

    const handleApplyHoming = () => {
        changePhase('homing');
        commands.startHoming();
    };

    const handleApplyUploadedCalibration = (payload: CalibrationUploadPayload) => {
        changePhase('done');
        commands.applyCalibration(payload);
    };

    // After homing result arrives, move to recording phase
    const homingDone = homingResult !== null;
    const isRecording = phase === 'recording';
    const calibrationDone = effectiveCalibration !== null;

    useEffect(() => {
        if (phase === 'homing' && homingDone && !recordingStarted) {
            setRecordingStarted(true);
            changePhase('recording');
            commands.startRecording();
        }
    }, [phase, homingDone, recordingStarted, changePhase, commands]);

    const handleStopRecording = () => {
        commands.stopRecording();
        changePhase('done');
    };

    if (error && (phase !== 'done' || !calibrationDone)) {
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

                    <CalibrationJsonImportCard onApply={handleApplyUploadedCalibration} />

                    <Flex justifyContent='space-between'>
                        <Button variant='secondary' onPress={goBack}>
                            Back
                        </Button>
                        <Button variant='accent' onPress={handleApplyHoming}>
                            Apply Homing Offsets
                        </Button>
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
                                            {Object.entries(effectiveCalibration.calibration).map(([name, cal]) => (
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
