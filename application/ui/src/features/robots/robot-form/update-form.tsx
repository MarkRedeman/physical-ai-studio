import type { FormEvent } from 'react';

import { Button, Divider, Flex, Form, View } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { paths } from '../../../paths';
import { SchemaRobotInput } from '../robot-types';
import { useRobotId } from '../use-robot';
import { FormFields, RobotFormHeading } from './form';
import { OfflineSaveButton, useRobotSubmission } from './live-connect/submission';
import { useRobotFormBody } from './provider';

import classes from './form.module.css';

export const UpdateRobotForm = () => {
    const navigate = useNavigate();
    const { robot_id, project_id } = useRobotId();

    const updateRobotMutation = $api.useMutation('put', '/api/projects/{project_id}/robots/{robot_id}', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/robots', { params: { path: { project_id } } }],
                ['get', '/api/projects/{project_id}/robots/{robot_id}', { params: { path: { project_id, robot_id } } }],
            ],
        },
    });

    const body = useRobotFormBody(robot_id);

    const save = async (snapshot: SchemaRobotInput) => {
        await updateRobotMutation.mutateAsync({
            params: { path: { project_id, robot_id } },
            body: snapshot,
        });

        navigate(paths.project.robots.show({ project_id, robot_id }));
    };
    const submission = useRobotSubmission(body, save);
    const handleSubmit = (event: FormEvent) => {
        event.preventDefault();
        void submission.submit();
    };

    return (
        <Flex direction='column' gap='size-200'>
            <RobotFormHeading heading='Update robot' />
            <Divider orientation='horizontal' size='S' />
            <Form onSubmit={handleSubmit}>
                <Flex direction='column' gap='size-200'>
                    <fieldset className={classes.fields} disabled={submission.pending}>
                        <Flex direction='column' gap='size-200' width='100%'>
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
                                Update robot
                            </Button>
                            <OfflineSaveButton
                                disabled={!submission.canSaveOffline}
                                onSave={() => void submission.submit(true)}
                            />
                        </Flex>
                    </View>
                </Flex>
            </Form>
        </Flex>
    );
};
