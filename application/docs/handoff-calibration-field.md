# CalibrationField Handoff

This note summarizes the new calibration upload flow added to Studio robot forms and what plugin authors should do to use it.

## What was added

- UI now includes a new `CalibrationField` component in `application/ui/src/features/robots/robot-form/robot-schema/components/calibration-field.tsx`.
- The schema form renderer (`schema-form.tsx`) now detects SO101-style calibration map fields and renders an upload control instead of skipping the raw object field.
- Users can upload a `.json` calibration file, Studio parses it, validates basic structure, and writes it into the payload field.
- For `BimanualSO101`, there are two independent upload controls:
  - `left_calibration`
  - `right_calibration`

## Expected file format

Upload JSON should match the existing SO101 export shape (joint-name keyed object):

```json
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 10,
        "range_min": -100,
        "range_max": 100
    }
}
```

For bimanual, upload two separate files (one per arm).

## How auto-detection works today

SchemaForm renders `CalibrationField` when all of these are true:

1. Field type is `object`.
2. Field name contains `calibration`.
3. Field is a map (`additionalProperties` present).
4. Entry schema is either untyped (`additionalProperties: true`) or SO101-like with integer fields:
   - `id`
   - `drive_mode`
   - `homing_offset`
   - `range_min`
   - `range_max`

This keeps existing plugin schemas working without introducing a new UI-kind contract.

## Plugin author guidance

### SO101 / Bimanual SO101

No plugin-side changes are required if payload fields already exist as:

- `calibration` for single-arm SO101
- `left_calibration` / `right_calibration` for bimanual SO101

Studio will now show upload controls and persist parsed JSON into those fields.

### LeKiwi (or other plugins)

If you want the same upload UX right now, define calibration payload fields as object maps named with `calibration` in the field name and keep the entry values typed as integer properties where possible.

If LeKiwi uses a different calibration schema (non-map or different field names/types), this first version may not auto-detect it. In that case, we should extend the detection rules or add an explicit plugin UI item kind later.

## Current limitations

- Upload-only flow (no inline JSON editor yet).
- Validation is structural and type-based; it does not perform robot-specific semantic checks.
- Detection is heuristic-based, not an explicit schema metadata kind.

## Files changed in this implementation

- `application/ui/src/features/robots/robot-form/robot-schema/components/calibration-field.tsx`
- `application/ui/src/features/robots/robot-form/robot-schema/schema-form.tsx`
- `application/ui/src/features/robots/robot-form/catalog/bimanual-so101.tsx`
- `application/ui/src/features/robots/robot-form/robot-schema/schema-form.test.tsx`
- `application/ui/src/features/robots/robot-form/catalog/bimanual-so101.test.tsx`
