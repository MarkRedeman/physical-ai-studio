import { Button, DialogTrigger, Text } from '@geti-ui/ui';

import { ImportModelDialog } from './import-model-dialog';

interface ImportModelButtonProps {
    buttonLabel?: string;
    onImportCompleted?: () => void;
}

export const ImportModelButton = ({
    buttonLabel = 'Import model',
    onImportCompleted,
}: ImportModelButtonProps = {}) => {
    return (
        <DialogTrigger>
            <Button variant='secondary'>
                <Text>{buttonLabel}</Text>
            </Button>

            {(close) => <ImportModelDialog onClose={close} onImportCompleted={onImportCompleted} />}
        </DialogTrigger>
    );
};

export { ImportModelDialog } from './import-model-dialog';
