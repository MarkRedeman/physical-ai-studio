import { useEffect, useRef, useState } from 'react';

import { Flex, ProgressCircle } from '@geti-ui/ui';

import { useFittedMediaSize } from '../../../features/cameras/use-fitted-media-size';
import { useInterval } from '../../../routes/datasets/use-interval';
import { useRobotControl } from '../robot-control-provider';

import classes from './camera-cell.module.css';

interface CameraDiagnostics {
    received: number;
    rendered: number;
    bytes: number;
    decodeMs: number;
}

const diagnosticsEnabled = () => new URLSearchParams(window.location.search).has('cameraDiagnostics');

const emptyDiagnostics = (): CameraDiagnostics => ({ received: 0, rendered: 0, bytes: 0, decodeMs: 0 });

export const CameraCell = ({ camera_id, camera_name }: { camera_id: string; camera_name: string }) => {
    const [img, setImg] = useState<string>();
    const [diagnostics, setDiagnostics] = useState<CameraDiagnostics>();
    const { observation } = useRobotControl();
    const diagnosticsRef = useRef<CameraDiagnostics>(emptyDiagnostics());
    const imageLoadStartedAt = useRef<number | undefined>(undefined);
    const lastPayload = useRef<string | undefined>(undefined);
    // TODO: Change hardcoding of fps and aspect ratio.
    // Not all camera types contain that info. Until a solution is found this is hardcoded.
    useInterval(() => {
        const id = camera_id;
        if (id !== undefined && observation.current?.cameras[id]) {
            const payload = observation.current.cameras[id];
            if (payload === lastPayload.current) {
                return;
            }
            lastPayload.current = payload;
            diagnosticsRef.current.received += 1;
            diagnosticsRef.current.bytes += payload.length;
            imageLoadStartedAt.current = performance.now();
            setImg(payload);
        }
    }, 1000 / 30);

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

    const imageRef = useRef<HTMLImageElement>(null);
    const { containerRef, width, height } = useFittedMediaSize(
        imageRef.current?.naturalWidth,
        imageRef.current?.naturalHeight
    );
    const kibPerFrame = diagnostics?.received ? (diagnostics.bytes / diagnostics.received / 1024).toFixed(1) : '0.0';

    const onImageLoad = () => {
        diagnosticsRef.current.rendered += 1;
        if (imageLoadStartedAt.current !== undefined) {
            diagnosticsRef.current.decodeMs += performance.now() - imageLoadStartedAt.current;
        }
    };

    return (
        <div ref={containerRef} className={classes.camera}>
            {img === undefined ? (
                <Flex width='100%' height='100%' justifyContent={'center'} alignItems={'center'}>
                    <ProgressCircle isIndeterminate />
                </Flex>
            ) : (
                <img
                    ref={imageRef}
                    alt={`Camera frame of ${camera_name}`}
                    src={`data:image/jpg;base64,${img}`}
                    onLoad={onImageLoad}
                    style={{
                        objectFit: 'contain',
                        height,
                        width,
                    }}
                />
            )}
            {diagnostics && (
                <div className={classes.diagnostics}>
                    <div>
                        rx {diagnostics.received} fps / render {diagnostics.rendered} fps
                    </div>
                    <div>
                        base64 {kibPerFrame} KiB/frame / decode{' '}
                        {diagnostics.rendered ? (diagnostics.decodeMs / diagnostics.rendered).toFixed(1) : '0.0'} ms
                    </div>
                </div>
            )}
        </div>
    );
};
