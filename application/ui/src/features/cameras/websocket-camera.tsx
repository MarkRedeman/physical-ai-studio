import { useCallback, useEffect, useRef, useState } from 'react';

import { Flex, ProgressCircle } from '@geti-ui/ui';
import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';
import { SchemaProjectCamera } from '../../api/types';
import { useFittedMediaSize } from './use-fitted-media-size';

import classes from './websocket-camera.module.css';

const CAMERA_WS_URL = fetchClient.PATH('/api/cameras/ws');

interface CameraDiagnostics {
    received: number;
    rendered: number;
    dropped: number;
    bytes: number;
    decodeMs: number;
}

const emptyDiagnostics = (): CameraDiagnostics => ({ received: 0, rendered: 0, dropped: 0, bytes: 0, decodeMs: 0 });

const CameraCanvas = ({
    camera,
    width,
    height,
    diagnosticsEnabled,
}: {
    camera: SchemaProjectCamera;
    width: number;
    height: number;
    diagnosticsEnabled: boolean;
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isLoading, setIsLoading] = useState(true);
    const processingRef = useRef(false);
    const frameQueueRef = useRef<Blob | null>(null);
    const diagnosticsRef = useRef<CameraDiagnostics>(emptyDiagnostics());
    const [diagnostics, setDiagnostics] = useState<CameraDiagnostics>();

    const processFrame = useCallback(async (blobData: Blob) => {
        diagnosticsRef.current.received += 1;
        diagnosticsRef.current.bytes += blobData.size;
        if (processingRef.current) {
            if (frameQueueRef.current !== null) {
                diagnosticsRef.current.dropped += 1;
            }
            frameQueueRef.current = blobData;
            return;
        }

        processingRef.current = true;
        try {
            const startedAt = performance.now();
            const bitmap = await createImageBitmap(blobData);
            diagnosticsRef.current.decodeMs += performance.now() - startedAt;
            const canvas = canvasRef.current;
            const ctx = canvas?.getContext('2d', { alpha: false });

            if (canvas && ctx) {
                ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
                setIsLoading(false);
                diagnosticsRef.current.rendered += 1;
            }

            bitmap.close();

            if (frameQueueRef.current) {
                const queuedBlob = frameQueueRef.current;
                frameQueueRef.current = null;
                processingRef.current = false;
                await processFrame(queuedBlob);
                return;
            }
        } catch (error) {
            console.error('Failed to process camera frame:', error);
        } finally {
            processingRef.current = false;
        }
    }, []);

    useEffect(() => {
        if (!diagnosticsEnabled) {
            return;
        }
        const interval = window.setInterval(() => {
            setDiagnostics(diagnosticsRef.current);
            diagnosticsRef.current = emptyDiagnostics();
        }, 1000);
        return () => window.clearInterval(interval);
    }, [diagnosticsEnabled]);

    // WebSocket message handler
    const handleMessage = useCallback(
        (event: WebSocketEventMap['message']) => {
            try {
                if (event.data instanceof Blob) {
                    // Binary JPEG frame
                    void processFrame(event.data);
                } else {
                    console.info('Received unknown event', event.data);
                }
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        },
        [processFrame]
    );

    useWebSocket(CAMERA_WS_URL, {
        queryParams: {
            camera: JSON.stringify({
                ...camera,
                // Prevent the stream from resetting anytime the user changes the camera name
                name: camera.hardware_name ?? '_',
            }),
        },
        share: true,
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });

    const kibPerFrame = diagnostics?.received ? (diagnostics.bytes / diagnostics.received / 1024).toFixed(1) : '0.0';

    return (
        <div className={classes.camera}>
            {isLoading && (
                <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                    <ProgressCircle isIndeterminate />
                </Flex>
            )}
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                style={{ display: isLoading ? 'none' : 'block' }}
                aria-label={`Camera: ${camera.name}`}
            />
            {diagnostics && (
                <div className={classes.diagnostics}>
                    <div>
                        rx {diagnostics.received} fps / render {diagnostics.rendered} fps
                    </div>
                    <div>
                        drop {diagnostics.dropped} / decode{' '}
                        {diagnostics.rendered ? (diagnostics.decodeMs / diagnostics.rendered).toFixed(1) : '0.0'} ms
                    </div>
                    <div>{kibPerFrame} KiB/frame</div>
                </div>
            )}
        </div>
    );
};

export const WebsocketCamera = ({
    camera,
    diagnosticsEnabled = false,
}: {
    camera: SchemaProjectCamera;
    diagnosticsEnabled?: boolean;
}) => {
    const { containerRef, width, height } = useFittedMediaSize(
        Number(camera.payload?.width),
        Number(camera.payload?.height)
    );

    return (
        <div ref={containerRef} style={{ height: '100%', width: '100%' }}>
            <CameraCanvas camera={camera} width={width} height={height} diagnosticsEnabled={diagnosticsEnabled} />
        </div>
    );
};
