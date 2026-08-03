import { useRef } from 'react';

import useWebSocket, { Options } from 'react-use-websocket';
import { v4 as uuidv4 } from 'uuid';

export default function useWebSocketWithResponse(
    url: string | (() => string | Promise<string>) | null,
    options?: Options,
    connect?: boolean
) {
    const messagePromises = useRef<Map<string, (message: MessageEvent) => void>>(new Map());
    const socket = useWebSocket(
        url,
        {
            ...options,
            onMessage: (event) => {
                for (const [_, callback] of messagePromises.current) {
                    callback(event);
                }
                if (options?.onMessage) {
                    options.onMessage(event);
                }
            },
        },
        connect
    );

    const sendJsonMessageAndWait = <MessageType>(
        data: object,
        matcher: (message: MessageType) => boolean,
        messageOptions?: { timeout: number }
    ): Promise<MessageType> => {
        const requestId = uuidv4();
        socket.sendJsonMessage(data);

        return new Promise((resolve, reject) => {
            messagePromises.current.set(requestId, (message) => {
                // Binary camera frames carry no JSON payload; skip them for
                // request/response matching so JSON.parse is never fed a Blob.
                if (typeof message.data !== 'string') {
                    return;
                }
                const messageData = JSON.parse(message.data) as MessageType;
                if (matcher(messageData)) {
                    messagePromises.current.delete(requestId);
                    resolve(messageData);
                }
            });
            if (messageOptions?.timeout)
                setTimeout(() => {
                    if (messagePromises.current.has(requestId)) {
                        messagePromises.current.delete(requestId);
                        reject(new Error('WebSocket request timed out.'));
                    }
                }, messageOptions?.timeout);
        });
    };

    return {
        ...socket,
        sendJsonMessageAndWait,
    };
}
