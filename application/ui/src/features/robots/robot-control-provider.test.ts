import { describe, expect, it } from 'vitest';

import { unpackCameraFrame } from './robot-control-provider';

const encodePayload = (cameraId: string, jpeg: Uint8Array): ArrayBuffer => {
    const idBytes = new TextEncoder().encode(cameraId);
    const payload = new Uint8Array(1 + idBytes.length + jpeg.length);
    payload.set([idBytes.length], 0);
    payload.set(idBytes, 1);
    payload.set(jpeg, 1 + idBytes.length);
    return payload.buffer;
};

describe('unpackCameraFrame', () => {
    it('splits the camera id header from the JPEG bytes', async () => {
        const jpeg = new Uint8Array([0xff, 0xd8, 0x01, 0x02]);
        const { cameraId, jpeg: blob } = unpackCameraFrame(encodePayload('cam-123', jpeg));

        expect(cameraId).toBe('cam-123');
        expect(blob.type).toBe('image/jpeg');
        expect(new Uint8Array(await blob.arrayBuffer())).toEqual(jpeg);
    });

    it('round-trips non-ascii camera ids', () => {
        const jpeg = new Uint8Array([0xff, 0xd8]);
        const { cameraId, jpeg: blob } = unpackCameraFrame(encodePayload("Bob's view", jpeg));

        expect(cameraId).toBe("Bob's view");
        expect(blob.size).toBe(2);
    });

    it('handles an empty payload body', async () => {
        const { cameraId, jpeg: blob } = unpackCameraFrame(encodePayload('cam', new Uint8Array(0)));

        expect(cameraId).toBe('cam');
        expect(await blob.arrayBuffer()).toEqual(new ArrayBuffer(0));
    });
});
