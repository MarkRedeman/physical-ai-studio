import { useEffect, useRef, useState } from 'react';

import { SchemaEpisode } from '../../../api/openapi-spec';

export interface Player {
    time: number;
    duration: number;
    isPlaying: boolean;
    seekId: number;
    play: () => void;
    pause: () => void;
    rewind: () => void;
    seek: (time: number) => void;
}

export const usePlayer = (episode: SchemaEpisode): Player => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [time, setTime] = useState<number>(0);
    const [seekId, setSeekId] = useState(0);
    const timeRef = useRef(0);
    const duration = episode.length / episode.fps;
    const frameTime = 1 / episode.fps;

    const setTimeSynced = (newTime: number) => {
        timeRef.current = newTime;
        setTime(newTime);
    };

    const play = () => {
        if (timeRef.current + frameTime > duration) {
            setTimeSynced(0);
            setSeekId((id) => id + 1);
        }
        setIsPlaying(true);
    };

    const pause = () => {
        setIsPlaying(false);
    };

    const rewind = () => {
        setTimeSynced(0);
        setSeekId((id) => id + 1);
    };

    const seek = (newTime: number) => {
        setTimeSynced(newTime);
        setSeekId((id) => id + 1);
    };

    useEffect(() => {
        setTimeSynced(0);
        setSeekId((id) => id + 1);
        setIsPlaying(false);
    }, [episode]);

    useEffect(() => {
        if (isPlaying) {
            const timeAtStart = timeRef.current;
            const worldTimeAtStart = new Date().getTime() / 1000;
            const interval = setInterval(() => {
                const now = new Date().getTime() / 1000;
                const nextTime = timeAtStart + now - worldTimeAtStart;
                if (nextTime > duration) {
                    setIsPlaying(false);
                }
                setTimeSynced(Math.min(nextTime, duration));
            }, frameTime * 1000);
            return () => clearInterval(interval);
        }
    }, [isPlaying, duration, frameTime, seekId]);

    return {
        time,
        duration,
        isPlaying,
        seekId,
        play,
        pause,
        rewind,
        seek,
    };
};
