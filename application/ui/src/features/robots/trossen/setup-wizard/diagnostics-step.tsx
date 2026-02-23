import {
    ActionButton,
    Button,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Heading,
    Icon,
    Loading,
    Text,
} from '@geti/ui';
import { Refresh } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { TrossenWizardStep, useTrossenSetupActions, useTrossenSetupState } from './wizard-provider';

import classes from '../../shared/setup-wizard/setup-wizard.module.scss';

/**
 * Trossen diagnostics step — shows IP reachability and driver configure status.
 *
 * Much simpler than SO101: no voltage check, no per-motor probe, no calibration
 * status. The Trossen SDK's `configure()` call validates the connection and
 * homes the robot as a side effect.
 */
export const TrossenDiagnosticsStep = () => {
    const { wsState } = useTrossenSetupState();
    const { goNext, markCompleted, commands } = useTrossenSetupActions();
    const navigate = useNavigate();

    const { diagnosticsResult, error } = wsState;
    const isLoading = !diagnosticsResult;

    if (error) {
        return (
            <Flex direction='column' gap='size-200'>
                <div className={classes.errorBox}>
                    <Text>
                        <strong>Connection Error:</strong> {error}
                    </Text>
                </div>
                <Flex gap='size-200'>
                    <Button variant='secondary' onPress={() => navigate(-1)}>
                        Back
                    </Button>
                </Flex>
            </Flex>
        );
    }

    if (isLoading) {
        return (
            <Flex direction='column' gap='size-300' alignItems='center' justifyContent='center' minHeight='size-3000'>
                <Loading mode='inline' />
                <Text>{wsState.statusMessage ?? 'Connecting to robot...'}</Text>
            </Flex>
        );
    }

    const { ip_reachable, configure_ok, motor_count, connection_string, error_message } = diagnosticsResult;
    const allOk = ip_reachable && configure_ok;

    return (
        <Flex direction='column' gap='size-200'>
            {/* Header with refresh */}
            <Flex alignItems='center' justifyContent='space-between'>
                <Heading level={4} margin={0}>
                    Diagnostics
                </Heading>
                <ActionButton isQuiet onPress={commands.reProbe} aria-label='Re-check'>
                    <Icon>
                        <Refresh />
                    </Icon>
                </ActionButton>
            </Flex>

            {/* IP Reachability section */}
            <Disclosure defaultExpanded={!ip_reachable} isQuiet>
                <DisclosureTitle UNSAFE_className={classes.disclosureHeader}>
                    <Flex alignItems='center' gap='size-100' width='100%'>
                        <Text UNSAFE_style={{ fontWeight: 600, fontSize: 14 }}>IP Reachability</Text>
                        <Flex flex alignItems='center' justifyContent='end'>
                            {ip_reachable ? (
                                <span className={`${classes.statusBadge} ${classes.statusOk}`}>
                                    {connection_string} reachable
                                </span>
                            ) : (
                                <span className={`${classes.statusBadge} ${classes.statusError}`}>Not reachable</span>
                            )}
                        </Flex>
                    </Flex>
                </DisclosureTitle>
                <DisclosurePanel>
                    <Flex direction='column' gap='size-100' marginTop='size-100'>
                        {ip_reachable ? (
                            <div className={classes.successBox}>
                                <Text>
                                    Robot at <strong>{connection_string}</strong> is reachable on the network.
                                </Text>
                            </div>
                        ) : (
                            <div className={classes.errorBox}>
                                <Text>
                                    Cannot reach <strong>{connection_string}</strong>. Verify the robot is powered on
                                    and connected to the same network, then re-check.
                                </Text>
                            </div>
                        )}
                    </Flex>
                </DisclosurePanel>
            </Disclosure>

            {/* Driver Configuration section — only shown when IP is reachable */}
            {ip_reachable && (
                <Disclosure defaultExpanded={!configure_ok} isQuiet>
                    <DisclosureTitle UNSAFE_className={classes.disclosureHeader}>
                        <Flex alignItems='center' gap='size-100' width='100%'>
                            <Text UNSAFE_style={{ fontWeight: 600, fontSize: 14 }}>Robot Connection</Text>
                            <Flex flex alignItems='center' justifyContent='end'>
                                {configure_ok ? (
                                    <span className={`${classes.statusBadge} ${classes.statusOk}`}>
                                        {motor_count} motors OK
                                    </span>
                                ) : (
                                    <span className={`${classes.statusBadge} ${classes.statusError}`}>
                                        Configuration failed
                                    </span>
                                )}
                            </Flex>
                        </Flex>
                    </DisclosureTitle>
                    <DisclosurePanel>
                        <Flex direction='column' gap='size-100' marginTop='size-100'>
                            {configure_ok ? (
                                <>
                                    <div className={classes.successBox}>
                                        <Text>
                                            Driver configured successfully. All {motor_count} motors are responding.
                                        </Text>
                                    </div>
                                    <div className={classes.infoBox}>
                                        <Text>
                                            The robot has been homed to its zero position as part of the connection
                                            process. This is normal for Trossen robots.
                                        </Text>
                                    </div>
                                </>
                            ) : (
                                <div className={classes.errorBox}>
                                    <Text>
                                        {error_message ??
                                            'Failed to configure the robot driver. ' +
                                                'Check that no other application is connected to the robot.'}
                                    </Text>
                                </div>
                            )}
                        </Flex>
                    </DisclosurePanel>
                </Disclosure>
            )}

            {/* Actions */}
            <Flex gap='size-200' justifyContent='space-between'>
                <Button variant='secondary' onPress={() => navigate(-1)}>
                    Back
                </Button>
                <Flex gap='size-200'>
                    {allOk && (
                        <Button
                            variant='accent'
                            onPress={() => {
                                markCompleted(TrossenWizardStep.DIAGNOSTICS);
                                commands.enterVerification();
                                goNext();
                            }}
                        >
                            Continue to Verification
                        </Button>
                    )}
                </Flex>
            </Flex>
        </Flex>
    );
};
