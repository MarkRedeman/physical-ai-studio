import type { FormEvent } from 'react';

import { Button, Divider, Flex, Form, View } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import type { SchemaSo101RobotPayload } from '../../../api/openapi-spec';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../paths';
import { useRobotCatalogQuery } from '../robot-catalog.hooks';
import { SchemaRobotInput } from '../robot-types';
import { FormFields, RobotFormHeading, RobotType } from './form';
import { OfflineSaveButton, useRobotSubmission } from './live-connect/submission';
import { useRobotForm, useRobotFormBody } from './provider';

import classes from './form.module.css';

const hasUploadedCalibration = (payload: unknown): boolean => {
    if (typeof payload !== 'object' || payload === null || Array.isArray(payload)) {
        return false;
    }
    const calibration = (payload as SchemaSo101RobotPayload).calibration;
    return typeof calibration === 'object' && calibration !== null && Object.keys(calibration).length > 0;
};

export const CreateRobotForm = () => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();
    const { activeType, draftId } = useRobotForm();
    const { data: catalog } = useRobotCatalogQuery();

    const addRobotMutation = $api.useMutation('post', '/api/projects/{project_id}/robots', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/robots', { params: { path: { project_id } } }],
                ['get', '/api/projects/{project_id}/robots/online', { params: { path: { project_id } } }],
            ],
        },
    });

    const body = useRobotFormBody(draftId);
    const isSO101 = activeType === 'SO101_Follower' || activeType === 'SO101_Leader';
    const shouldRunSO101Setup = isSO101 && !hasUploadedCalibration(body?.payload);
    const shouldRunZeroCalibration = catalog.some(
        ({ type, zero_calibration }) => type === activeType && zero_calibration != null
    );
    const zeroCalibrationPath =
        activeType === 'ReBot_B601_RS_Follower'
            ? paths.project.robots.rebotRsSetup
            : activeType === 'StarArm_102_LD_Leader' || activeType === 'StarArm_102_HD_Leader'
              ? paths.project.robots.stararmSetup
              : paths.project.robots.zeroCalibration;

    const save = async (snapshot: SchemaRobotInput) => {
        if (shouldRunSO101Setup) {
            navigate(paths.project.robots.so101Setup({ project_id }));
            return;
        }

        if (shouldRunZeroCalibration) {
            navigate(zeroCalibrationPath({ project_id }));
            return;
        }

        const createdRobot = await addRobotMutation.mutateAsync({
            params: { path: { project_id } },
            body: snapshot,
        });

        navigate(paths.project.robots.show({ project_id, robot_id: createdRobot.id }));
    };
    const submission = useRobotSubmission(body, save, shouldRunSO101Setup || shouldRunZeroCalibration);
    const handleSubmit = (event: FormEvent) => {
        event.preventDefault();
        void submission.submit();
    };

    return (
        <Flex direction='column' gap='size-200'>
            <RobotFormHeading heading='Configure new robot' />
            <Divider orientation='horizontal' size='S' />
            <Form onSubmit={handleSubmit}>
                <Flex direction='column' gap='size-200'>
                    <fieldset className={classes.fields} disabled={submission.pending}>
                        <Flex direction='column' gap='size-200' width='100%'>
                            <RobotType />
                            <FormFields />
                        </Flex>
                    </fieldset>
                    <Divider orientation='horizontal' size='S' />
                    {submission.error !== null && <div role='alert'>{submission.error}</div>}
                    <View>
                        <Flex gap='size-100' wrap='wrap'>
                            <Button
                                variant='accent'
                                type='submit'
                                isDisabled={!submission.canSubmit}
                                isPending={submission.pending}
                            >
                                {shouldRunSO101Setup
                                    ? 'Begin Setup'
                                    : shouldRunZeroCalibration
                                      ? 'Begin Calibration'
                                      : 'Add robot'}
                            </Button>
                            {!shouldRunSO101Setup && !shouldRunZeroCalibration && (
                                <OfflineSaveButton
                                    disabled={!submission.canSaveOffline}
                                    onSave={() => void submission.submit(true)}
                                />
                            )}
                        </Flex>
                    </View>
                </Flex>
            </Form>
        </Flex>
    );
};
