# Importing Datasets

This guide explains the user-visible dataset import flow in the UI.

## Start import
[//]: # (Screenshot suggestion: Import dataset dialog in upload step with Dataset name field and ZIP drop zone.)

In Datasets:

1. Open the add/import menu.
2. Choose Import.
3. Enter Dataset name.
4. Upload a ZIP archive.

The upload step starts an asynchronous import job in the background.

## What users see next
[//]: # (Screenshot suggestion: in-progress import state showing detection/processing message.)

After upload, the UI moves through stages automatically:

- Detection/in-progress view.
- User review view (when metadata is ready).
- Importing/completion view.

## User review step
[//]: # (Screenshot suggestion: user review step showing detected dataset summary, environment picker, task field, and Finalize import button.)

In review, users can typically see:

- Dataset summary (episodes, detected robots/cameras).
- Recording environment selector.
- Optional task field.

To continue, users click Finalize import.

## If detection fails
[//]: # (Screenshot suggestion: detection failed view with validation details list and Cancel action.)

The UI shows a failure message and any available details.

From a user point of view:

- Close the dialog.
- Confirm the archive is correct.
- Retry import.

## Practical expectations
[//]: # (Screenshot suggestion: completed import where the new dataset appears in dataset tabs/list.)

- Import is not instant for large archives.
- Keep the UI open until the job reaches completion.
- Imported datasets appear in dataset tabs when done.

## Next
[//]: # (Screenshot suggestion: optional docs links screenshot to Exporting Datasets and Dataset Management chapters.)

- Export flow: `application/docs/061-exporting-datasets.md`.
- Dataset management overview: `application/docs/06-dataset-management.md`.
