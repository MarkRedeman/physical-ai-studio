import { useSuspenseQuery } from '@tanstack/react-query';

export type CatalogEntry = {
    type: string;
    display_name: string;
    role: string;
    urdf_path?: string | null;
};

export const useCatalog = () => {
    return useSuspenseQuery<CatalogEntry[]>({
        queryKey: ['robot-catalog'],
        queryFn: () => fetch('/api/robots/catalog').then((r) => r.json()),
    });
};

export const useCatalogEntry = (type: string) => {
    return useSuspenseQuery<CatalogEntry>({
        queryKey: ['robot-catalog', type],
        queryFn: () => fetch(`/api/robots/catalog/${type}`).then((r) => r.json()),
    });
};
