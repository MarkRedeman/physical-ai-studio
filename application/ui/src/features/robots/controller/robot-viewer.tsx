/* eslint-disable react/no-unknown-property */

import { Suspense, useEffect, useMemo, useRef } from 'react';

import { ContactShadows, Grid, OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import { degToRad } from 'three/src/math/MathUtils.js';
import { URDFRobot } from 'urdf-loader';

import { useContainerSize } from '../../../components/zoom/use-container-size';
import { useRobotCatalogDefinitionQuery } from '../robot-catalog.hooks';
import { SchemaRobot } from '../robot-types';
import { mapJointToURDFJoint, useLoadModelQuery } from './../robot-models-context';

import classes from './robot-viewer.module.css';

/** Material name used by the dark parts in the Trossen URDF. */
const TROSSEN_DARK_MATERIAL = 'trossen_black';

const SCENE_COLORS = {
    background: '#242528',
    ambientLight: '#c8d6eb',
    primaryLight: '#e8f0ff',
    fillLight: '#88aadd',
    gridCell: '#3a3d4f',
    gridSection: '#545870',
    checkerboardEven: '#282a30',
    checkerboardOdd: '#2c2e34',
    trossenReplacement: new THREE.Color('#585858'),
};

const FLOOR_SIZE = 21;
const CHECKERBOARD_TILE_SIZE = 0.5;

/**
 * Find the shared `trossen_black` material on the model and replace its dark
 * texture with a solid color.
 *
 * The model is guaranteed to have all its STL meshes loaded before it enters
 * React state (see `useLoadModelQuery` which resolves on
 * `LoadingManager.onLoad`), so a plain `useEffect` is sufficient here.
 *
 * Because urdf-loader uses a shared material instance for each named material,
 * mutating it in-place ensures all meshes (even nested deep in the tree) pick
 * up the change.  Originals are restored on cleanup.
 */
const useBrightenDarkMaterials = (model: URDFRobot | undefined, enabled: boolean) => {
    useEffect(() => {
        if (!model || !enabled) return;

        const saved: {
            mat: THREE.MeshPhongMaterial;
            map: THREE.Texture | null;
            color: THREE.Color;
        }[] = [];

        const seen = new Set<THREE.Material>();

        model.traverse((node) => {
            if (!(node as THREE.Mesh).isMesh) {
                return;
            }
            const mesh = node as THREE.Mesh;
            const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];

            for (const mat of materials) {
                if (seen.has(mat)) {
                    continue;
                }

                seen.add(mat);

                if (!mat.name.toLowerCase().includes(TROSSEN_DARK_MATERIAL)) {
                    continue;
                }

                const phong = mat as THREE.MeshPhongMaterial;
                saved.push({ mat: phong, map: phong.map, color: phong.color.clone() });

                phong.map = null;
                phong.color.copy(SCENE_COLORS.trossenReplacement);
                phong.needsUpdate = true;
            }
        });

        return () => {
            for (const s of saved) {
                s.mat.map = s.map;
                s.mat.color.copy(s.color);
                s.mat.needsUpdate = true;
            }
        };
    }, [model, enabled]);
};

const useConfigureModelShadows = (model: URDFRobot) => {
    useEffect(() => {
        model.traverse((node) => {
            if ((node as THREE.Mesh).isMesh) {
                (node as THREE.Mesh).castShadow = true;
            }
        });
    }, [model]);
};

const CheckerboardFloor = () => {
    const texture = useMemo(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;

        const context = canvas.getContext('2d');
        if (!context) {
            return null;
        }

        const tiles = 24;
        const tileSize = canvas.width / tiles;
        for (let x = 0; x < tiles; x += 1) {
            for (let y = 0; y < tiles; y += 1) {
                context.fillStyle = (x + y) % 2 === 0 ? SCENE_COLORS.checkerboardEven : SCENE_COLORS.checkerboardOdd;
                context.fillRect(x * tileSize, y * tileSize, tileSize, tileSize);
            }
        }

        const checkerboardTexture = new THREE.CanvasTexture(canvas);
        checkerboardTexture.wrapS = THREE.RepeatWrapping;
        checkerboardTexture.wrapT = THREE.RepeatWrapping;
        checkerboardTexture.repeat.set(
            FLOOR_SIZE / (tiles * CHECKERBOARD_TILE_SIZE),
            FLOOR_SIZE / (tiles * CHECKERBOARD_TILE_SIZE)
        );

        return checkerboardTexture;
    }, []);

    useEffect(() => {
        return () => texture?.dispose();
    }, [texture]);

    return (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.005, 0]} receiveShadow>
            <planeGeometry args={[FLOOR_SIZE, FLOOR_SIZE]} />
            <meshStandardMaterial map={texture} roughness={0.8} metalness={0} />
        </mesh>
    );
};

// This is a wrapper component for the loaded URDF model
const ActualURDFModel = ({ model, isTrossen }: { model: URDFRobot; isTrossen: boolean }) => {
    // Rotate -90 degrees around X-axis (π/2 radians)
    const rotation = [-Math.PI / 2, 0, (-1 * Math.PI) / 4] as const;
    const scale = [3, 3, 3] as const;

    useBrightenDarkMaterials(model, isTrossen);
    useConfigureModelShadows(model);

    return (
        <group rotation={rotation} scale={scale}>
            <primitive object={model} />
        </group>
    );
};

interface RobotViewerProps {
    robot: Pick<SchemaRobot, 'type'>;
    featureValues?: number[];
    featureNames?: string[];
}
export const RobotViewer = ({ robot = { type: 'SO101_Follower' }, featureValues, featureNames }: RobotViewerProps) => {
    const angle = degToRad(-45);
    const isTrossen = robot.type.toLowerCase().includes('trossen');

    const { data: definition } = useRobotCatalogDefinitionQuery(robot.type);
    const jointMap = definition.joint_map;

    const { data: model, error, isPending } = useLoadModelQuery(robot.type);
    const ref = useRef<HTMLDivElement>(null);
    const size = useContainerSize(ref);

    useEffect(() => {
        if (featureValues !== undefined && featureNames !== undefined && model !== undefined) {
            featureNames.forEach((_, index) => {
                mapJointToURDFJoint(
                    {
                        name: featureNames[index],
                        value: featureValues[index],
                    },
                    model,
                    jointMap
                );
            });
        }
    }, [featureValues, featureNames, model, jointMap]);

    return (
        <div ref={ref} className={classes.viewer}>
            <div className={classes.canvas} style={{ height: `${size.height}px`, width: `${size.width}px` }}>
                <Canvas>
                    <color attach='background' args={[SCENE_COLORS.background]} />
                    <ambientLight intensity={0.7} color={SCENE_COLORS.ambientLight} />
                    <directionalLight
                        position={[-1.5, 3.5, 2]}
                        intensity={1.5}
                        color={SCENE_COLORS.primaryLight}
                        castShadow
                        shadow-mapSize-width={2048}
                        shadow-mapSize-height={2048}
                        shadow-camera-left={-3 * 2}
                        shadow-camera-right={3 * 2}
                        shadow-camera-top={3 * 2}
                        shadow-camera-bottom={-3 * 2}
                        shadow-camera-near={0.1 * 2}
                        shadow-camera-far={20 * 2}
                        shadow-bias={-0.0001}
                    />
                    <directionalLight position={[2, 2, -3]} intensity={0.4} color={SCENE_COLORS.fillLight} />
                    <PerspectiveCamera makeDefault position={[2.0, 1, 1]} />
                    <OrbitControls enableDamping={false} />
                    <CheckerboardFloor />
                    <Grid
                        infiniteGrid
                        cellSize={0.25}
                        cellColor={SCENE_COLORS.gridCell}
                        sectionSize={0.5}
                        sectionColor={SCENE_COLORS.gridSection}
                        fadeDistance={FLOOR_SIZE - 1}
                    />
                    <ContactShadows
                        position={[0, 0, 0]}
                        opacity={0.2}
                        scale={2.5}
                        blur={2.5}
                        far={1}
                        resolution={512}
                    />
                    {model && (
                        <group key={model.uuid} position={[0, 0, 0]} rotation={[0, angle, 0]}>
                            <Suspense fallback={null}>
                                <ActualURDFModel model={model} isTrossen={isTrossen} />
                            </Suspense>
                        </group>
                    )}
                </Canvas>
                {isPending && <div className={classes.loadingOverlay}>Loading robot model...</div>}
                {error && (
                    <div className={classes.errorOverlay} role='alert'>
                        Failed to load robot model: {error instanceof Error ? error.message : String(error)}
                    </div>
                )}
            </div>
        </div>
    );
};
