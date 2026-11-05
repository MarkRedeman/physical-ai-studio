import { Suspense } from 'react';

import { Grid, Loading, minmax, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaEnvironmentDraft, SchemaModel } from '../../../api/openapi-spec';
import { AppFooter } from '../../../components/app-footer/app-footer';
import { paths } from '../../../paths';
import { environmentDraft, RuntimeSessionProvider } from '../../runtime-session/runtime-session-provider';
import { SessionWorkspace } from '../../runtime-session/workspace/session-workspace';
import { useInferenceParams } from './use-inference-params';

const InferenceSession = ({ model }: { model: SchemaModel }) => (
    <SessionWorkspace
        initialMode='inference'
        cancelHref={paths.project.models.index({ project_id: model.project_id })}
    />
);

/** The model running in a session, on the environment it was trained for if it knows it. */
const InferenceWorkspace = ({
    model,
    environment,
}: {
    model: SchemaModel;
    environment: SchemaEnvironmentDraft | null;
}) => {
    const { project_id, backend, device, variant } = useInferenceParams();

    return (
        <RuntimeSessionProvider
            project_id={project_id}
            environment={environment}
            model={{ model_id: model.id!, backend, device: device ?? 'auto', variant }}
        >
            <Grid
                areas={['content', 'footer']}
                UNSAFE_style={{
                    gridTemplateRows: `${minmax(0, '1fr')} var(--spectrum-global-dimension-size-400)`,
                }}
                minHeight={0}
                height={'100%'}
            >
                <View gridArea={'content'} padding='size-200' minHeight={0} height='100%'>
                    {/* Inside the provider: anything that suspends while loading must not take the
                            page-level boundary, which would unmount the provider and end the session. */}
                    <Suspense fallback={<Loading mode='inline' />}>
                        <InferenceSession model={model} />
                    </Suspense>
                </View>
                <AppFooter />
            </Grid>
        </RuntimeSessionProvider>
    );
};

/** The environment of the dataset the model was trained on. */
const TrainedOn = ({ model, dataset_id }: { model: SchemaModel; dataset_id: string }) => {
    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: { path: { dataset_id } },
    });
    const { data: environment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        { params: { path: { project_id: model.project_id, environment_id: dataset.environment_id } } }
    );

    return <InferenceWorkspace model={model} environment={environmentDraft(environment)} />;
};

export const InferencePage = () => {
    const { model_id } = useInferenceParams();
    const {
        data: { model },
    } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id } },
    });

    // A model without a dataset (imported) starts with no environment; the side bar loads one.
    return model.dataset_id ? (
        <TrainedOn model={model} dataset_id={model.dataset_id} />
    ) : (
        <InferenceWorkspace model={model} environment={null} />
    );
};
