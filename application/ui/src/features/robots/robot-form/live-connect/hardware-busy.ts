import { createContext, useContext } from 'react';

// Schema fields also render outside the live form (tests and setup). This small
// optional context avoids coupling them to the connection lifecycle provider.
export const LiveConnectHardwareBusyContext = createContext(false);
export const useLiveConnectHardwareBusy = () => useContext(LiveConnectHardwareBusyContext);
