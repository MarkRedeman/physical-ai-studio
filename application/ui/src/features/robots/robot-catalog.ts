import { useQuery, useSuspenseQuery } from '@tanstack/react-query';

import { SchemaRobotType } from './robot-types';

export type CatalogEntry = {
    type: string;
    display_name: string;
    role: string;
    urdf_path?: string | null;
    package_map?: Record<string, string>;
    joint_map?: Record<string, string[]>;
    asset_source?: string;
};

const fetchCatalog = (): Promise<CatalogEntry[]> =>
    fetch('/api/robots/catalog').then((r) => {
        if (!r.ok) throw new Error('Failed to fetch catalog');
        return r.json();
    });

export const useCatalog = () => {
    return useSuspenseQuery<CatalogEntry[]>({
        queryKey: ['robot-catalog'],
        queryFn: fetchCatalog,
    });
};

export const useCatalogEntry = (type: string) => {
    return useSuspenseQuery<CatalogEntry>({
        queryKey: ['robot-catalog', type],
        queryFn: () => fetch(`/api/robots/catalog/${type}`).then((r) => r.json()),
    });
};

const useCatalogData = () => {
    return useQuery<CatalogEntry[]>({
        queryKey: ['robot-catalog'],
        queryFn: fetchCatalog,
        staleTime: 60_000,
    });
};

export const useUrdfPathForType = (robotType: SchemaRobotType): string => {
    const { data } = useCatalogData();
    const entry = data?.find((e) => e.type === robotType);
    return entry?.urdf_path ?? '';
};

export const usePackageMapForType = (robotType: SchemaRobotType): Record<string, string> => {
    const { data } = useCatalogData();
    const entry = data?.find((e) => e.type === robotType);
    return entry?.package_map ?? {};
};

export const useJointMapForType = (robotType: SchemaRobotType): Record<string, string[]> => {
    const { data } = useCatalogData();
    const entry = data?.find((e) => e.type === robotType);
    return entry?.joint_map ?? {};
};

export const useRobotRoleChecks = () => {
    const { data } = useCatalogData();

    const isFollower = (type: string): boolean => {
        return data?.some((e) => e.type === type && e.role === 'follower') ?? false;
    };

    const isLeader = (type: string): boolean => {
        return data?.some((e) => e.type === type && e.role === 'leader') ?? false;
    };

    return { isFollower, isLeader };
};
