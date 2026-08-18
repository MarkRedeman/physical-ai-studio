export type RobotUiInfo = {
    title?: string;
    text: string;
    variant?: 'info' | 'warning';
};

export type ConnectionGroupOptions = {
    title?: string;
    description?: string;
    identify?: boolean;
    connection_key?: string;
    serial_number_key?: string;
    manual_entry?: boolean;
    infos?: RobotUiInfo[];
};

export type GroupOptions = ConnectionGroupOptions & {
    device_discovery?: boolean;
};

export type FieldOptions = {
    group?: string;
    widget?: 'device-selector';
    required?: boolean;
};

export type ModelUiOptions = {
    groups?: Record<string, GroupOptions>;
    infos?: RobotUiInfo[];
};

export type FieldSchema = {
    type?: string;
    title?: string;
    description?: string;
    default?: unknown;
    enum?: unknown[];
    $ref?: string;
    properties?: Record<string, FieldSchema>;
    additionalProperties?: FieldSchema | boolean;
    required?: string[];
    ['x-physicalai-ui']?: FieldOptions & ModelUiOptions;
};

export type JsonSchema = {
    type?: string;
    properties?: Record<string, FieldSchema>;
    required?: string[];
    $defs?: Record<string, FieldSchema>;
    ['x-physicalai-ui']?: ModelUiOptions;
};
