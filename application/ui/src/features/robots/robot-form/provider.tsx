import { createContext, ReactNode, useCallback, useContext, useMemo, useState } from 'react';

import { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../robot-types';
import { getInitialSO101FormData } from './catalog/so101';
import { getInitialWidowxFormData } from './catalog/widowxai';
import { getInitialBimanualFormData } from './catalog/widowxai-bimanual';
import { buildRobotBody, typeForSchema, type AnyRobotFormData, type RobotType, type RobotTypeData } from './form-data';

type RobotFormState = {
    activeType: SchemaRobotType;
    so101: RobotTypeData['so101'];
    widowx: RobotTypeData['widowx'];
    bimanual_widowx: RobotTypeData['bimanual_widowx'];
};

const RobotFormContext = createContext<RobotFormState | null>(null);

type SetRobotFormContextType = {
    setActiveType: (type: SchemaRobotType) => void;
    updateFormData: <R extends RobotType>(
        robotType: R,
        update: Partial<RobotTypeData[R]> | ((prev: RobotTypeData[R]) => RobotTypeData[R])
    ) => void;
};

const SetRobotFormContext = createContext<SetRobotFormContextType | null>(null);

const getInitialState = (robot?: SchemaRobot): RobotFormState => {
    const activeType = robot?.type ?? 'SO101_Follower';
    const robotType = typeForSchema[activeType];

    return {
        activeType,
        so101: getInitialSO101FormData(
            robotType === 'so101' && robot
                ? {
                      name: robot.name,
                      serial_number: robot.payload.serial_number,
                      connection_string: 'connection_string' in robot.payload ? robot.payload.connection_string : '',
                  }
                : undefined
        ),
        widowx: getInitialWidowxFormData(
            robotType === 'widowx' && robot
                ? {
                      name: robot.name,
                      connection_string: 'connection_string' in robot.payload ? robot.payload.connection_string : '',
                      serial_number: robot.payload.serial_number,
                  }
                : undefined
        ),
        bimanual_widowx: getInitialBimanualFormData(
            robotType === 'bimanual_widowx' && robot
                ? {
                      name: robot.name,
                      connection_string_left:
                          'connection_string_left' in robot.payload ? robot.payload.connection_string_left : '',
                      connection_string_right:
                          'connection_string_right' in robot.payload ? robot.payload.connection_string_right : '',
                      serial_number: robot.payload.serial_number,
                  }
                : undefined
        ),
    };
};

export const RobotFormProvider = ({ children, robot }: { children: ReactNode; robot?: SchemaRobot }) => {
    const [state, setState] = useState(() => getInitialState(robot));

    const setActiveType = useCallback((type: SchemaRobotType) => {
        setState((prev) => ({ ...prev, activeType: type }));
    }, []);

    const updateFormData = useCallback(
        <R extends RobotType>(
            robotType: R,
            update: Partial<RobotTypeData[R]> | ((prev: RobotTypeData[R]) => RobotTypeData[R])
        ) => {
            setState((prev) => ({
                ...prev,
                [robotType]:
                    typeof update === 'function'
                        ? (update as (prev: RobotTypeData[R]) => RobotTypeData[R])(prev[robotType])
                        : { ...prev[robotType], ...update },
            }));
        },
        []
    );

    const setContextValue = useMemo(() => ({ setActiveType, updateFormData }), [setActiveType, updateFormData]);

    return (
        <RobotFormContext.Provider value={state}>
            <SetRobotFormContext.Provider value={setContextValue}>{children}</SetRobotFormContext.Provider>
        </RobotFormContext.Provider>
    );
};

export function useRobotForm(): { activeType: SchemaRobotType; robotForm: AnyRobotFormData };
export function useRobotForm<R extends RobotType>(
    robotType: R
): { activeType: SchemaRobotType; robotForm: RobotTypeData[R] };
export function useRobotForm(robotType?: RobotType) {
    const context = useContext(RobotFormContext);

    if (context === null) {
        throw new Error('useRobotForm was used outside of RobotFormProvider');
    }

    const rt = robotType ?? typeForSchema[context.activeType];

    return { activeType: context.activeType, robotForm: context[rt] };
}

export const useSetRobotForm = () => {
    const context = useContext(SetRobotFormContext);

    if (context === null) {
        throw new Error('useSetRobotForm was used outside of RobotFormProvider');
    }

    return context;
};

export const useRobotFormFields = <R extends RobotType>(robotType: R) => {
    const state = useContext(RobotFormContext);
    const { updateFormData } = useSetRobotForm();

    if (state === null) {
        throw new Error('useRobotFormFields was used outside of RobotFormProvider');
    }

    const updateField = <K extends keyof RobotTypeData[R]>(field: K, value: RobotTypeData[R][K]) => {
        updateFormData(robotType, { [field]: value } as unknown as Partial<RobotTypeData[R]>);
    };

    return { formData: state[robotType] as RobotTypeData[R], activeType: state.activeType, updateField };
};

export const useRobotFormBody = (robot_id: string): SchemaRobotInput | null => {
    const state = useContext(RobotFormContext);

    if (state === null) {
        return null;
    }

    const robotType = typeForSchema[state.activeType];
    const formData = state[robotType] as AnyRobotFormData;

    return buildRobotBody(robotType, formData, state.activeType, robot_id);
};
