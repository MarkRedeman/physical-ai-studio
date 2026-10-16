export type HotkeyActionId =
    | 'recording.start_episode'
    | 'recording.start_reverse_replay'
    | 'recording.accept_episode'
    | 'recording.discard_episode';

type HotkeyActionDefinition = {
    label: string;
    scope: string;
    defaultCombo: string;
};

export const HOTKEY_ACTIONS: Record<HotkeyActionId, HotkeyActionDefinition> = {
    'recording.start_episode': {
        label: 'Start episode',
        scope: 'Recording',
        defaultCombo: 'ArrowRight',
    },
    'recording.start_reverse_replay': {
        label: 'Record reverse replay',
        scope: 'Recording',
        defaultCombo: 'ArrowUp',
    },
    'recording.accept_episode': {
        label: 'Accept episode',
        scope: 'Recording',
        defaultCombo: 'ArrowRight',
    },
    'recording.discard_episode': {
        label: 'Discard episode',
        scope: 'Recording',
        defaultCombo: 'ArrowLeft',
    },
};

export const HOTKEY_ACTION_IDS = Object.keys(HOTKEY_ACTIONS) as HotkeyActionId[];
