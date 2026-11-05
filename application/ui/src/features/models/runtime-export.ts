export const runtimeExportUrl = ({
    modelId,
    environmentId,
    backend,
    device,
    task,
    variant,
    executionHorizon,
    maxGuidanceWeight,
}: {
    modelId: string;
    environmentId: string;
    backend: string;
    device: string;
    task?: string;
    variant?: string;
    executionHorizon?: number;
    maxGuidanceWeight?: number;
}): string => {
    const params = new URLSearchParams({
        environment_id: environmentId,
        device,
    });
    if (task !== undefined && task.trim() !== '') {
        params.set('task', task);
    }
    if (variant !== undefined) params.set('variant', variant);
    if (executionHorizon !== undefined) params.set('execution_horizon', String(executionHorizon));
    if (maxGuidanceWeight !== undefined) params.set('max_guidance_weight', String(maxGuidanceWeight));
    return `/api/models/${modelId}/exports/${backend}/download?${params.toString()}`;
};
