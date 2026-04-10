import { useEffect, useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { isAbortError } from '../../utils/download';
import type { FormatHint } from './use-dataset-import-job-state';

interface UseDatasetUploadResult {
    upload: (file: File, formatHint: FormatHint, datasetName: string) => Promise<string | undefined>;
    abort: () => void;
    progress: number | null;
    mutation: ReturnType<
        typeof useMutation<{ id: string }, Error, { file: File; source: string; datasetName: string }>
    >;
}

export const useDatasetUpload = (projectId: string): UseDatasetUploadResult => {
    const [progress, setProgress] = useState<number | null>(null);
    const abortRef = useRef<XMLHttpRequest | null>(null);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
        };
    }, []);

    const mutation = useMutation({
        mutationFn: async ({ file, source, datasetName }: { file: File; source: string; datasetName: string }) => {
            const { data: preparedJob, error } = await fetchClient.POST(
                '/api/projects/{project_id}/imports/datasets:prepare',
                {
                    params: { path: { project_id: projectId } },
                    body: { format_hint: source, dataset_name: datasetName },
                    bodySerializer: (body) => {
                        const fd = new FormData();
                        const b = body as { format_hint: string; dataset_name: string };
                        fd.append('format_hint', b.format_hint);
                        fd.append('dataset_name', b.dataset_name);
                        return fd;
                    },
                }
            );

            if (error || !preparedJob) {
                throw new Error('Failed to prepare dataset import job');
            }

            const jobId = (preparedJob as { id?: string }).id;
            if (!jobId) {
                throw new Error('Failed to prepare import: missing job id');
            }

            const uploadPath = fetchClient.PATH('/api/projects/{project_id}/imports/datasets/{job_id}:upload', {
                params: { path: { project_id: projectId, job_id: jobId } },
            });

            return await new Promise<{ id: string }>((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                abortRef.current = xhr;
                xhr.open('PUT', uploadPath);
                xhr.responseType = 'json';

                xhr.upload.onprogress = (event) => {
                    if (event.lengthComputable && event.total > 0) {
                        setProgress(Math.round((event.loaded / event.total) * 100));
                    } else {
                        setProgress(null);
                    }
                };

                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        const uploaded = xhr.response as { id?: string } | null;
                        resolve({ id: uploaded?.id ?? jobId });
                    } else {
                        reject(new Error(`Failed to upload dataset archive: ${xhr.status}`));
                    }
                };

                xhr.onerror = () => reject(new Error('Failed to upload dataset archive'));
                xhr.onabort = () => reject(new DOMException('Upload aborted', 'AbortError'));

                const fd = new FormData();
                fd.append('archive', file);
                xhr.send(fd);
            });
        },
        onMutate: () => {
            setProgress(null);
        },
        onSettled: () => {
            abortRef.current = null;
        },
    });

    const upload = async (file: File, formatHint: FormatHint, datasetName: string): Promise<string | undefined> => {
        try {
            const job = await mutation.mutateAsync({ file, source: formatHint, datasetName });
            return job.id;
        } catch (error) {
            if (isAbortError(error)) {
                return undefined;
            }
            return undefined;
        }
    };

    const abort = () => {
        abortRef.current?.abort();
        mutation.reset();
        setProgress(null);
    };

    return { upload, abort, progress, mutation };
};
