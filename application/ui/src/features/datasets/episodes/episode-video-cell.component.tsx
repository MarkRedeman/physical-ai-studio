import { useEffect, useRef } from 'react';

import { fetchClient } from '../../../api/client';
import { SchemaEpisodeVideo } from '../../../api/openapi-spec';
import { useFittedMediaSize } from '../../cameras/use-fitted-media-size';
import { useEpisodeViewer } from './episode-viewer-provider.component';

export const EpisodeVideoCell = ({
    episodeVideo,
    datasetId,
}: {
    episodeVideo: SchemaEpisodeVideo;
    datasetId: string;
}) => {
    const { player } = useEpisodeViewer();
    const url = fetchClient.PATH('/api/dataset/{dataset_id}/video/{video_path}', {
        params: {
            path: {
                dataset_id: datasetId,
                video_path: episodeVideo.path,
            },
        },
    });

    const videoRef = useRef<HTMLVideoElement>(null);

    // Seek and play/pause on explicit player state changes only — not on every time tick.
    // `seekId` increments when the user seeks/rewinds; `isPlaying` changes on play/pause.
    // Reading `player.time` inside the effect is safe: React batches the state updates
    // that trigger seekId/isPlaying changes together with the time update.
    useEffect(() => {
        const video = videoRef.current;
        const start = episodeVideo.start;
        if (!video || !Number.isFinite(start)) return;

        video.currentTime = player.time + start;
        if (player.isPlaying) {
            video.play();
        } else {
            video.pause();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [player.isPlaying, player.seekId, episodeVideo.start]);

    const { containerRef, width, height } = useFittedMediaSize(
        videoRef.current?.videoWidth,
        videoRef.current?.videoHeight
    );

    /* eslint-disable jsx-a11y/media-has-caption */
    return (
        <div ref={containerRef} style={{ height: '100%', width: '100%' }}>
            <video ref={videoRef} src={url} width={width} height={height} />
        </div>
    );
};
