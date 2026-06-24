import { createContext, useContext } from 'react';

export const RobotFormElementContext = createContext<string | null>(null);

export const useRobotFormElement = () => {
    return useContext(RobotFormElementContext);
};
