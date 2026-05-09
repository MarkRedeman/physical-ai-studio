import { Loading, Text, useNumberFormatter, View } from '@geti-ui/ui';

import classes from './number-badge.module.css';

interface NumberBadgeProps {
    jobsNumber: number | null;
    isPending?: boolean;
    isAccented?: boolean;
    isSelected?: boolean;
}

const getNumberClasses = (number: number): string => {
    const size = number >= 100 ? 'large' : number >= 10 ? 'medium' : '';

    if (size) {
        return `${classes.number} ${classes[size]}`;
    }

    return classes.number;
};

export const NumberBadge = ({ jobsNumber, isPending, isSelected = false, isAccented = false }: NumberBadgeProps) => {
    const formatter = useNumberFormatter({ notation: 'compact' });

    if (isPending || jobsNumber === null) {
        return <Loading mode='inline' size={'S'} />;
    }

    return (
        <>
            {jobsNumber === 0 ? (
                <></>
            ) : (
                <View
                    borderRadius={'large'}
                    width={'size-200'}
                    height={'size-200'}
                    data-testid='number badge'
                    UNSAFE_className={`${classes.badge} ${classes.circle} ${
                        isAccented ? classes.accented : isSelected ? classes.selected : classes.basic
                    }`}
                >
                    <Text UNSAFE_className={getNumberClasses(jobsNumber)}>{formatter.format(jobsNumber)}</Text>
                </View>
            )}
        </>
    );
};
