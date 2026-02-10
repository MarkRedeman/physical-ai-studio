import {
    SchemaBaslerCameraInput,
    SchemaGenicamCameraInput,
    SchemaIpCameraInput,
    SchemaRealsenseCameraInput,
    SchemaUsbCameraInput,
    SchemaWebsocketCameraInput,
} from './openapi-spec';

//
export type SchemaProjectCamera =
    | SchemaUsbCameraInput
    | SchemaBaslerCameraInput
    | SchemaGenicamCameraInput
    | SchemaIpCameraInput
    | SchemaRealsenseCameraInput
    | SchemaUsbCameraInput
    | SchemaWebsocketCameraInput;
