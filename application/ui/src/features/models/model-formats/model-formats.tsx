import { Suspense, useState } from 'react';

import { DialogContainer, Grid, Loading } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { SchemaModel, SchemaModelVariantAction } from '../../../api/openapi-spec';
import { isExportBackend } from '../inference-backends';
import { BackendCard } from './backend-card';
import { CreateVariantDialog } from './create-variant-dialog';

import classes from './model-formats.module.css';

const ModelFormatsContents = ({ model }: { model: SchemaModel }) => {
    const { data: detail } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id: model.id! } },
    });
    const [creation, setCreation] = useState<{ initialAction?: SchemaModelVariantAction } | null>(null);
    const actions = detail.available_actions ?? [];
    return (
        <>
            <Grid
                gap='size-200'
                marginTop='size-400'
                UNSAFE_style={{
                    gridTemplateColumns:
                        'repeat(auto-fill, minmax(min(100%, var(--spectrum-global-dimension-size-4600)), 1fr))',
                }}
            >
                {detail.exports.flatMap((backend) => {
                    const type = backend.type;
                    return isExportBackend(type)
                        ? (backend.variants ?? []).map((variant) => (
                              <BackendCard
                                  key={`${type}/${variant.id}`}
                                  backendType={type}
                                  model={detail.model}
                                  variant={variant}
                                  action={
                                      variant.id === 'original'
                                          ? actions.find(
                                                (action) => action.backend === type && action.operation === 'int8_sym'
                                            )
                                          : undefined
                                  }
                                  onCreate={(initialAction) => setCreation({ initialAction })}
                              />
                          ))
                        : [];
                })}
                {actions.length > 0 && (
                    <button
                        type='button'
                        aria-label='Add variant'
                        className={classes.addCard}
                        onClick={() => setCreation({})}
                    >
                        <Add aria-hidden='true' />
                        <span className={classes.addTitle}>Add variant</span>
                        <span>Optimize or export to another backend</span>
                    </button>
                )}
            </Grid>
            <DialogContainer onDismiss={() => setCreation(null)}>
                {creation !== null && (
                    <CreateVariantDialog
                        modelDetail={detail}
                        initialAction={creation.initialAction}
                        close={() => setCreation(null)}
                    />
                )}
            </DialogContainer>
        </>
    );
};

export const ModelFormats = ({ model }: { model: SchemaModel }) => (
    <Suspense fallback={<Loading mode='inline' size='M' marginY='size-400' />}>
        <ModelFormatsContents model={model} />
    </Suspense>
);
