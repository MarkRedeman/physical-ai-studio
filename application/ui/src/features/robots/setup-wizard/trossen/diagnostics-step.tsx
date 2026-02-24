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

import { InlineAlert } from '../shared/inline-alert';
import { StatusBadge } from '../shared/status-badge';
import { TrossenWizardStep, useTrossenSetupActions, useTrossenSetupState } from './wizard-provider';

import classes from '../shared/setup-wizard.module.scss';

/**
 * Trossen diagnostics step — shows IP reachability and driver configure status.
 *
 * Much simpler than SO101: no voltage check, no per-motor probe, no calibration
 * status. The Trossen SDK's `configure()` call validates the connection and
 * homes the robot as a side effect.
 */
export const TrossenDiagnosticsStep = () => {
    const { wsState } = useTrossenSetupState();
    const { goNext, markCompleted, onBackToRobotInfo, commands } = useTrossenSetupActions();

    const { diagnosticsResult, error } = wsState;
    const isLoading = !diagnosticsResult;

    if (error) {
        return (
            <Flex direction='column' gap='size-200'>
                <InlineAlert variant='error'>
                    <strong>Connection Error:</strong> {error}
                </InlineAlert>
                <Flex gap='size-200'>
                    <Button variant='secondary' onPress={onBackToRobotInfo}>
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
                                <StatusBadge variant='ok'>{connection_string} reachable</StatusBadge>
                            ) : (
                                <StatusBadge variant='error'>Not reachable</StatusBadge>
                            )}
                        </Flex>
                    </Flex>
                </DisclosureTitle>
                <DisclosurePanel>
                    <Flex direction='column' gap='size-100' marginTop='size-100'>
                        {ip_reachable ? (
                            <InlineAlert variant='success'>
                                Robot at <strong>{connection_string}</strong> is reachable on the network.
                            </InlineAlert>
                        ) : (
                            <InlineAlert variant='error'>
                                Cannot reach <strong>{connection_string}</strong>. Verify the robot is powered on and
                                connected to the same network, then re-check.
                            </InlineAlert>
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
                                    <StatusBadge variant='ok'>{motor_count} motors OK</StatusBadge>
                                ) : (
                                    <StatusBadge variant='error'>Configuration failed</StatusBadge>
                                )}
                            </Flex>
                        </Flex>
                    </DisclosureTitle>
                    <DisclosurePanel>
                        <Flex direction='column' gap='size-100' marginTop='size-100'>
                            {configure_ok ? (
                                <>
                                    <InlineAlert variant='success'>
                                        Driver configured successfully. All {motor_count} motors are responding.
                                    </InlineAlert>
                                    <InlineAlert variant='info'>
                                        The robot has been homed to its zero position as part of the connection process.
                                        This is normal for Trossen robots.
                                    </InlineAlert>
                                </>
                            ) : (
                                <InlineAlert variant='error'>
                                    {error_message ??
                                        'Failed to configure the robot driver. ' +
                                            'Check that no other application is connected to the robot.'}
                                </InlineAlert>
                            )}
                        </Flex>
                    </DisclosurePanel>
                </Disclosure>
            )}

            {/* Actions */}
            <Flex gap='size-200' justifyContent='space-between'>
                <Button variant='secondary' onPress={onBackToRobotInfo}>
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
