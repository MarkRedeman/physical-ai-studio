import { Suspense } from 'react';

import { Grid, Loading, minmax, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { AppFooter } from '../../../components/app-footer/app-footer';
import { useDatasetId } from '../../../features/datasets/use-dataset';
import { environmentDraft, RuntimeSessionProvider } from '../../../features/runtime-session/runtime-session-provider';
import { BackToDatasetButton } from '../../../features/runtime-session/workspace/back-to-dataset-button';
import { SessionWorkspace } from '../../../features/runtime-session/workspace/session-workspace';
import { paths } from '../../../paths';

const RecordingPage = () => {
    const { project_id, dataset_id } = useDatasetId();
    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: { path: { dataset_id } },
    });
    const { data: environment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            params: { path: { environment_id: dataset.environment_id, project_id } },
        }
    );

    return (
        <RuntimeSessionProvider
            project_id={project_id}
            environment={environmentDraft(environment)}
            dataset_id={dataset_id}
        >
            <Grid
                areas={['content', 'footer']}
                UNSAFE_style={{ gridTemplateRows: `${minmax(0, '1fr')} var(--spectrum-global-dimension-size-400)` }}
                minHeight={0}
                height='100%'
            >
                <View gridArea='content' padding='size-200' minHeight={0} height='100%'>
                    {/* Keep suspension inside the provider so loading UI does not end the session. */}
                    <Suspense fallback={<Loading mode='inline' />}>
                        <SessionWorkspace
                            initialMode='record'
                            cancelHref={paths.project.datasets.show({ project_id, dataset_id })}
                            backButton={<BackToDatasetButton dataset_id={dataset_id} />}
                        />
                    </Suspense>
                </View>
                <AppFooter />
            </Grid>
        </RuntimeSessionProvider>
    );
};

export const Index = () => (
    <Suspense fallback={<Loading mode='overlay' />}>
        <RecordingPage />
    </Suspense>
);
