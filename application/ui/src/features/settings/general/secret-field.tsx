import { useState } from 'react';

import { Flex, Switch, TextField } from '@geti-ui/ui';

export type SecretChange = {
    /** User's typed replacement value (empty when untouched). */
    draft: string;
    /** True when the user asked to revoke the configured secret. */
    remove: boolean;
};

type SecretFieldProps = {
    label: string;
    isSet: boolean;
    isDisabled?: boolean;
    onChange: (change: SecretChange) => void;
};

/**
 * Password input for an optional secret (e.g. an API token).
 *
 * The server never returns the plaintext, so the field starts empty. When a
 * secret is configured the placeholder explains that leaving the field empty
 * keeps the existing value, and a switch revokes it. The parent builds its
 * PATCH body from {@link SecretChange}.
 */
export const SecretField = ({ label, isSet, isDisabled = false, onChange }: SecretFieldProps) => {
    const [draft, setDraft] = useState('');
    const [remove, setRemove] = useState(false);

    const update = (nextDraft: string, nextRemove: boolean) => {
        setDraft(nextDraft);
        setRemove(nextRemove);
        onChange({ draft: nextDraft, remove: nextRemove });
    };

    return (
        <Flex direction='column' gap='size-100'>
            <TextField
                type='password'
                label={label}
                value={draft}
                onChange={(value) => update(value, remove)}
                placeholder={isSet ? 'Leave empty to keep the configured value' : undefined}
                isDisabled={isDisabled}
                width='100%'
            />
            {isSet && (
                <Switch isSelected={remove} isDisabled={isDisabled} onChange={(selected) => update(draft, selected)}>
                    Remove the configured value
                </Switch>
            )}
        </Flex>
    );
};
