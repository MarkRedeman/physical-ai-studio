import { useCallback, useEffect, useRef, useState } from 'react';

import { Flex, ProgressCircle } from '@geti-ui/ui';

import { useFittedMediaSize } from '../../../features/cameras/use-fitted-media-size';
import { useRobotControl } from '../robot-control-provider';

import classes from './camera-cell.module.css';

interface CameraDiagnostics {
    received: number;
    rendered: number;
    dropped: number;
    bytes: number;
    decodeMs: number;
}

const diagnosticsEnabled = () => new URLSearchParams(window.location.search).has('cameraDiagnostics');

const emptyDiagnostics = (): CameraDiagnostics => ({ received: 0, rendered: 0, dropped: 0, bytes: 0, decodeMs: 0 });

export const CameraCell = ({ camera_id, camera_name }: { camera_id: string; camera_name: string }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | undefined>();
    const processingRef = useRef(false);
    const frameQueueRef = useRef<Blob | null>(null);
    const diagnosticsRef = useRef<CameraDiagnostics>(emptyDiagnostics());
    const [diagnostics, setDiagnostics] = useState<CameraDiagnostics>();
    const { subscribeCamera } = useRobotControl();

    const drawFrame = useCallback(async (jpeg: Blob) => {
        diagnosticsRef.current.received += 1;
        diagnosticsRef.current.bytes += jpeg.size;
        // If a decode is in flight, keep only the newest frame so a slow
        // client always renders the latest frame and drops stale ones.
        if (processingRef.current) {
            if (frameQueueRef.current !== null) {
                diagnosticsRef.current.dropped += 1;
            }
            frameQueueRef.current = jpeg;
            return;
        }

        processingRef.current = true;
        try {
            const startedAt = performance.now();
            const bitmap = await createImageBitmap(jpeg);
            diagnosticsRef.current.decodeMs += performance.now() - startedAt;
            setNaturalSize((prevSize) => {
                if (prevSize && prevSize.width === bitmap.width && prevSize.height === bitmap.height) {
                    return prevSize;
                }
                return { width: bitmap.width, height: bitmap.height };
            });
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
                await drawFrame(queuedBlob);
                return;
            }
        } catch (error) {
            console.error('Failed to decode camera frame:', error);
        } finally {
            processingRef.current = false;
        }
    }, []);

    useEffect(() => subscribeCamera(camera_id, drawFrame), [camera_id, drawFrame, subscribeCamera]);

    useEffect(() => {
        if (!diagnosticsEnabled()) {
            return;
        }
        const interval = window.setInterval(() => {
            setDiagnostics(diagnosticsRef.current);
            diagnosticsRef.current = emptyDiagnostics();
        }, 1000);
        return () => window.clearInterval(interval);
    }, []);

    const { containerRef, width, height } = useFittedMediaSize(naturalSize?.width, naturalSize?.height);
    const kibPerFrame = diagnostics?.received ? (diagnostics.bytes / diagnostics.received / 1024).toFixed(1) : '0.0';

    return (
        <div ref={containerRef} className={classes.camera}>
            {isLoading && (
                <Flex width='100%' height='100%' justifyContent={'center'} alignItems={'center'}>
                    <ProgressCircle isIndeterminate />
                </Flex>
            )}
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                style={{ display: isLoading ? 'none' : 'block', objectFit: 'contain' }}
                aria-label={`Camera frame of ${camera_name}`}
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
