import { useCallback, useEffect, useRef, useState } from 'react';

import { Flex, ProgressCircle } from '@geti-ui/ui';

import { useFittedMediaSize } from '../../../features/cameras/use-fitted-media-size';
import { useRobotControl } from '../robot-control-provider';

export const CameraCell = ({ camera_id, camera_name }: { camera_id: string; camera_name: string }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | undefined>();
    const processingRef = useRef(false);
    const frameQueueRef = useRef<Blob | null>(null);
    const { subscribeCamera } = useRobotControl();

    const drawFrame = useCallback(async (jpeg: Blob) => {
        // If a decode is in flight, keep only the newest frame so a slow
        // client always renders the latest frame and drops stale ones.
        if (processingRef.current) {
            frameQueueRef.current = jpeg;
            return;
        }

        processingRef.current = true;
        try {
            const bitmap = await createImageBitmap(jpeg);
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

    const { containerRef, width, height } = useFittedMediaSize(naturalSize?.width, naturalSize?.height);

    return (
        <div ref={containerRef} style={{ height: '100%', width: '100%' }}>
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
        </div>
    );
};
