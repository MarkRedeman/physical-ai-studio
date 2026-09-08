# CalibrationField Handoff

This note summarizes the new calibration upload flow added to Studio robot forms and what plugin authors should do to use it.

## What was added

- UI now includes a new `CalibrationField` component in `application/ui/src/features/robots/robot-form/robot-schema/components/calibration-field.tsx`.
- The schema form renderer (`schema-form.tsx`) renders an upload control for fields explicitly declared with a `{"kind": "calibration"}` item in `x-physicalai-ui` metadata.
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

## How rendering works

SchemaForm renders `CalibrationField` only when the payload UI metadata includes a dedicated item:

```python
robot_payload_ui([
    {"kind": "calibration", "name": "calibration"},
])
```

This matches how other first-party controls are rendered (`connection`, `ip_address`) and avoids heuristic auto-detection.

## Plugin author guidance

### SO101 / Bimanual SO101

No plugin-side changes are required if payload fields already exist as:

- `calibration` for single-arm SO101
- `left_calibration` / `right_calibration` for bimanual SO101

Studio will now show upload controls and persist parsed JSON into those fields.

### LeKiwi (or other plugins)

If you want the same upload UX, add a `{"kind": "calibration", "name": "..."}` item to `robot_payload_ui(...)` for the payload field that should receive uploaded calibration JSON.

## Current limitations

- Upload-only flow (no inline JSON editor yet).
- Validation is structural and type-based; it does not perform robot-specific semantic checks.
- The control renders only for fields explicitly declared through a `calibration` UI item.

## Files changed in this implementation

- `application/ui/src/features/robots/robot-form/robot-schema/components/calibration-field.tsx`
- `application/ui/src/features/robots/robot-form/robot-schema/schema-form.tsx`
- `application/ui/src/features/robots/robot-form/catalog/bimanual-so101.tsx`
- `application/ui/src/features/robots/robot-form/robot-schema/schema-form.test.tsx`
- `application/ui/src/features/robots/robot-form/catalog/bimanual-so101.test.tsx`
