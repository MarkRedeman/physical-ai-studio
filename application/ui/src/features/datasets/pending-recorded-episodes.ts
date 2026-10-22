// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const PENDING_RECORDED_EPISODES_KEY_PREFIX = 'physicalai.pending-recorded-episodes';
const PENDING_RECORDED_EPISODES_TTL_MS = 60_000;

export type PendingRecordedEpisodes = {
    expectedEpisodeCount: number;
    expiresAt: number;
};

const getStorageKey = (dataset_id: string) => `${PENDING_RECORDED_EPISODES_KEY_PREFIX}:${dataset_id}`;

export const setPendingRecordedEpisodes = (dataset_id: string, expectedEpisodeCount: number) => {
    window.sessionStorage.setItem(
        getStorageKey(dataset_id),
        JSON.stringify({
            expectedEpisodeCount,
            expiresAt: Date.now() + PENDING_RECORDED_EPISODES_TTL_MS,
        } satisfies PendingRecordedEpisodes)
    );
};

export const getPendingRecordedEpisodes = (dataset_id: string): PendingRecordedEpisodes | undefined => {
    const storageKey = getStorageKey(dataset_id);
    const stored = window.sessionStorage.getItem(storageKey);
    if (stored === null) {
        return undefined;
    }

    try {
        const pending = JSON.parse(stored) as PendingRecordedEpisodes;
        if (
            typeof pending.expectedEpisodeCount !== 'number' ||
            !Number.isFinite(pending.expectedEpisodeCount) ||
            typeof pending.expiresAt !== 'number' ||
            pending.expiresAt <= Date.now()
        ) {
            window.sessionStorage.removeItem(storageKey);
            return undefined;
        }
        return pending;
    } catch {
        window.sessionStorage.removeItem(storageKey);
        return undefined;
    }
};

export const clearPendingRecordedEpisodes = (dataset_id: string) => {
    window.sessionStorage.removeItem(getStorageKey(dataset_id));
};
