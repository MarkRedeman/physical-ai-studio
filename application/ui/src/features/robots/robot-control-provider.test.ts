import { describe, expect, it } from 'vitest';

import { unpackCameraFrame } from './robot-control-provider';

const CAMERA_ID_FRAME_LENGTH = 36;

const encodePayload = (cameraId: string, jpeg: Uint8Array): ArrayBuffer => {
    const idBytes = new TextEncoder().encode(cameraId);
    expect(idBytes).toHaveLength(CAMERA_ID_FRAME_LENGTH);
    const payload = new Uint8Array(CAMERA_ID_FRAME_LENGTH + jpeg.length);
    payload.set(idBytes, 0);
    payload.set(jpeg, CAMERA_ID_FRAME_LENGTH);
    return payload.buffer;
};

describe('unpackCameraFrame', () => {
    it('splits the fixed-width camera id header from the JPEG bytes', async () => {
        const jpeg = new Uint8Array([0xff, 0xd8, 0x01, 0x02]);
        const { cameraId, jpeg: blob } = unpackCameraFrame(encodePayload('0823b0fd-5c9f-4c1a-9dd1-5f4e105aebe9', jpeg));

        expect(cameraId).toBe('0823b0fd-5c9f-4c1a-9dd1-5f4e105aebe9');
        expect(blob.type).toBe('image/jpeg');
        expect(new Uint8Array(await blob.arrayBuffer())).toEqual(jpeg);
    });

    it('handles an empty payload body', async () => {
        const cameraId = '0823b0fd-5c9f-4c1a-9dd1-5f4e105aebe9';
        const { cameraId: parsedId, jpeg: blob } = unpackCameraFrame(encodePayload(cameraId, new Uint8Array(0)));

        expect(parsedId).toBe(cameraId);
        expect(await blob.arrayBuffer()).toEqual(new ArrayBuffer(0));
    });
});
