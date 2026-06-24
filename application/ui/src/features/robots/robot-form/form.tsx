import { useRef, type FormEvent } from 'react';

import { Button, Divider, Flex, Form, Heading, Icon, Item, Picker, TextField } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { SchemaRobotType } from '../robot-types';
import { useRobotForm, useSetRobotForm } from './provider';
import { BiManualWidowxAIFormFields } from './catalog/widowxai-bimanual';
import { SO101FormFields } from './catalog/so101';
import { WidowxAIFormFields } from './catalog/widowxai';
import { SubmitNewRobotButton } from './submit-new-robot-button';

const RobotType = () => {
    const setRobotForm = useSetRobotForm();
    const robotForm = useRobotForm();

    return (
        <Picker
            isRequired
            label='Robot type'
            width='100%'
            selectedKey={robotForm.type}
            onSelectionChange={(selected) => {
                const newType = selected as typeof robotForm.type;

                const wasSerial = robotForm.type?.toLowerCase().startsWith('so101') ?? false;
                const isSerial = newType?.toLowerCase().startsWith('so101') ?? false;

                setRobotForm((oldForm) => ({
                    ...oldForm,
                    type: newType,
                    ...(wasSerial !== isSerial
                        ? {
                              // Only reset when switching families (SO -> Trossen, etc)
                              serial_number: '',
                              connection_string: '',
                              connection_string_left: '',
                              connection_string_right: '',
                          }
                        : {}),
                }));
            }}
        >
            <Item key={'SO101_Follower'}>SO101 Follower</Item>
            <Item key={'SO101_Leader'}>SO101 Leader</Item>
            <Item key={'Trossen_WidowXAI_Follower'}>Trossen WidowX AI Follower</Item>
            <Item key={'Trossen_WidowXAI_Leader'}>Trossen WidowX AI Leader</Item>
            <Item key={'Trossen_Bimanual_WidowXAI_Follower'}>Trossen Bimanual WidowX AI Follower</Item>
            <Item key={'Trossen_Bimanual_WidowXAI_Leader'}>Trossen Bimanual WidowX AI Leader</Item>
        </Picker>
    );
};

const FormFields = ({ robotType }: { robotType: SchemaRobotType }) => {
    switch (robotType) {
        case 'SO101_Follower':
        case 'SO101_Leader':
            return <SO101FormFields />;
        case 'Trossen_WidowXAI_Follower':
        case 'Trossen_WidowXAI_Leader':
            return <WidowxAIFormFields />;
        case 'Trossen_Bimanual_WidowXAI_Leader':
        case 'Trossen_Bimanual_WidowXAI_Follower':
            return <BiManualWidowxAIFormFields />;
    }
};

export const RobotForm = ({ heading = 'Add new robot', submitButton = <SubmitNewRobotButton /> }) => {
    const { project_id } = useProjectId();

    const robotForm = useRobotForm();
    const setRobotForm = useSetRobotForm();

    const submitContainerRef = useRef<HTMLDivElement>(null);

    // Make Enter behave like clicking the submit button. A disabled button ignores
    // the click, so validation still applies.
    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        submitContainerRef.current?.querySelector('button')?.click();
    };

    return (
        <Flex direction='column' gap='size-200'>
            <Flex alignItems={'center'} gap='size-200'>
                <Button
                    href={paths.project.robots.index({ project_id })}
                    variant='secondary'
                    UNSAFE_style={{ border: 'none' }}
                >
                    <Icon>
                        <ChevronLeft color='white' fill='white' />
                    </Icon>
                </Button>

                <Heading>{heading}</Heading>
            </Flex>
            <Divider orientation='horizontal' size='S' />
            {/* Prevent native form submission, which reloads the page and discards form state.
                Advancing to the next step is handled by the submit button's onPress. */}
            <Form onSubmit={handleSubmit}>
                <Flex direction='column' gap='size-200'>
                    <Flex direction='column' gap='size-200' width='100%'>
                        <TextField
                            isRequired
                            label='Robot name'
                            width='100%'
                            onChange={(name) => {
                                setRobotForm((oldForm) => ({ ...oldForm, name }));
                            }}
                            value={robotForm.name}
                        />

                        {/* Put robot type first as we can use it to visualize the robot
                          and determine how to connect with it */}
                        <RobotType />

                        <FormFields robotType={robotForm.type} />
                    </Flex>
                    <Divider orientation='horizontal' size='S' />
                    <div ref={submitContainerRef}>{submitButton}</div>
                </Flex>
            </Form>
        </Flex>
    );
};
