# Robot Plugin Catalog Plan

## Goal

Make robot catalog entries rich enough for Studio to present a scalable robot
selection experience while allowing plugins without a URDF.

## Catalog Contract

Every `RobotCatalogDefinition` provides:

- `category`: a human-readable grouping label, such as `ReBot`, `LeRobot`, or
  `SO101`.
- `source`: `internal`, `first_party`, or `external`. Studio uses this to let
  users hide externally supplied integrations.
- `asset`: an optional `RobotAsset`, including an optional `preview_thumbnail`
  image path. Entries without an asset can be configured
  and used normally; Studio does not offer a 3D preview for them.

## Implementation Steps

1. Add catalog metadata to the plugin SDK and document the public contract.
2. Set metadata for Studio-owned catalog definitions. Preserve compatibility
   for older entry-point plugins with neutral defaults.
3. Return category, source, thumbnail, and nullable URDF metadata from the
   catalog API.
4. Update the 3D preview to avoid loading a model when a catalog entry has no
   URDF.
5. Replace the flat robot-type picker with a modal catalog browser that:
   - Groups cards by category.
   - Filters by leader/follower role.
   - Can hide external integrations.
   - Displays supplied thumbnails with a visual fallback.
6. Cover the API contract and selection/filtering behavior with tests, then
   regenerate the UI OpenAPI types.

## Compatibility

External catalog plugins are loaded independently from Studio. Definitions
created against an earlier SDK receive `Other` as their category and `external`
as their source until the plugin is updated. This avoids preventing a Studio
upgrade from starting while still allowing plugin authors to opt into the full
catalog UI.
