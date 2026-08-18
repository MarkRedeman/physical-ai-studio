export type RobotUiInfo = {
    title?: string;
    text: string;
    variant?: 'info' | 'warning';
};

export type RobotUiConnectionBinding = {
    connection: string;
    serial_number?: string;
};

export type ConnectionControlOptions = {
    kind: 'connection';
    label?: string;
    description?: string;
    device_discovery?: boolean;
    identify?: boolean;
    manual_entry?: boolean;
    infos?: RobotUiInfo[];
    bind: RobotUiConnectionBinding;
};

export type ControlOptions = ConnectionControlOptions;

export type SectionOptions = {
    id: string;
    title?: string;
    description?: string;
    infos?: RobotUiInfo[];
    fields?: string[];
    controls?: ControlOptions[];
};

export type FieldOptions = {
    required?: boolean;
};

export type ModelUiOptions = {
    infos?: RobotUiInfo[];
    sections?: SectionOptions[];
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
    ['x-physicalai-ui']?: FieldOptions & Partial<ModelUiOptions>;
};

export type JsonSchema = {
    type?: string;
    properties?: Record<string, FieldSchema>;
    required?: string[];
    $defs?: Record<string, FieldSchema>;
    ['x-physicalai-ui']?: ModelUiOptions;
};
