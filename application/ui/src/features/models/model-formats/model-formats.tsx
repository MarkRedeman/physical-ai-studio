import { Suspense, useState } from 'react';

import { DialogContainer, Grid, Loading } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import type { SchemaModel } from '../../../api/openapi-spec';
import { isExportBackend } from '../inference-backends';
import { BackendCard } from './backend-card';
import { CreateVariantDialog } from './create-variant-dialog';

import classes from './model-formats.module.css';

interface ModelExportsProps {
    model: SchemaModel;
}

const cardGridColumns = 'repeat(auto-fill, minmax(min(100%, var(--spectrum-global-dimension-size-4600)), 1fr))';

const ModelFormatsContents = ({ model }: { model: SchemaModel }) => {
    const { data: modelDetail } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id: model.id! } },
    });
    const [creation, setCreation] = useState<{ sourceId?: string } | null>(null);
    const { data: jobs = [] } = $api.useQuery('get', '/api/jobs');
    const activeJobs = jobs.filter(
        (job) =>
            job.type === 'optimize_model' &&
            job.payload.model_id === model.id &&
            (job.status === 'pending' || job.status === 'running')
    );
    const hasExports = modelDetail.exports.some(
        (backend) => isExportBackend(backend.type) && backend.variants.length > 0
    );

    return (
        <>
            <Grid
                gap='size-200'
                marginTop='size-400'
                UNSAFE_style={{
                    gridTemplateColumns: cardGridColumns,
                }}
            >
                {modelDetail.exports.flatMap((backend) => {
                    const backendType = backend.type;
                    return isExportBackend(backendType)
                        ? backend.variants.map((variant) => (
                              <BackendCard
                                  key={`${backendType}/${variant.id}`}
                                  backendType={backendType}
                                  model={model}
                                  variant={variant}
                                  onOptimize={() => setCreation({ sourceId: `${backendType}/${variant.id}` })}
                              />
                          ))
                        : [];
                })}
                <button
                    type='button'
                    aria-label='Add variant'
                    className={classes.addCard}
                    onClick={() => setCreation({})}
                    disabled={!hasExports}
                >
                    <Add aria-hidden='true' />
                    <span className={classes.addTitle}>Add variant</span>
                    <span>
                        {hasExports
                            ? 'Optimize or re-export an existing model export'
                            : 'A complete backend export is required'}
                    </span>
                    {activeJobs.length > 0 && (
                        <span role='status'>
                            {activeJobs.length} variant{' '}
                            {activeJobs.length === 1 ? 'job in progress' : 'jobs in progress'}
                        </span>
                    )}
                </button>
            </Grid>
            <Suspense fallback={null}>
                <DialogContainer onDismiss={() => setCreation(null)}>
                    {creation !== null && (
                        <CreateVariantDialog
                            key={creation.sourceId ?? 'new'}
                            modelDetail={modelDetail}
                            initialSourceId={creation.sourceId}
                            close={() => setCreation(null)}
                        />
                    )}
                </DialogContainer>
            </Suspense>
        </>
    );
};

export const ModelFormats = ({ model }: ModelExportsProps) => {
    return (
        <Suspense fallback={<Loading mode='inline' size='M' marginY='size-400' />}>
            <ModelFormatsContents model={model} />
        </Suspense>
    );
};
