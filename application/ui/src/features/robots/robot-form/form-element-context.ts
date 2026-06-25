import { createContext, useContext } from 'react';

export const RobotFormElementContext = createContext<string | null>(null);

export const useRobotFormElement = () => {
    const context = useContext(RobotFormElementContext);

    if (context === null) {
        throw new Error('useRobotFormElement was used outside of RobotForm provider form context');
    }

    return document.getElementById(context) as HTMLFormElement | null;
};
