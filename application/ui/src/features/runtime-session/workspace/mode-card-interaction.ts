import { MouseEvent, PointerEvent, useRef } from 'react';

type CardClick = Pick<
    MouseEvent<HTMLElement>,
    | 'currentTarget'
    | 'target'
    | 'defaultPrevented'
    | 'button'
    | 'detail'
    | 'altKey'
    | 'ctrlKey'
    | 'metaKey'
    | 'shiftKey'
>;
type SurfaceClick = CardClick & Pick<MouseEvent<HTMLElement>, 'clientX' | 'clientY'>;
type SurfacePress = Pick<
    PointerEvent<HTMLElement>,
    | 'currentTarget'
    | 'target'
    | 'clientX'
    | 'clientY'
    | 'defaultPrevented'
    | 'button'
    | 'isPrimary'
    | 'altKey'
    | 'ctrlKey'
    | 'metaKey'
    | 'shiftKey'
>;

const INTERACTIVE_TARGETS = [
    'button',
    'input',
    'textarea',
    'select',
    'a[href]',
    'label',
    'summary',
    'form',
    '[tabindex]',
    '[contenteditable]',
    '[data-mode-interactive]',
    '[role="button"]',
    '[role="link"]',
    '[role="radio"]',
    '[role="radiogroup"]',
    '[role="checkbox"]',
    '[role="switch"]',
    '[role="slider"]',
    '[role="spinbutton"]',
    '[role="textbox"]',
    '[role="searchbox"]',
    '[role="combobox"]',
    '[role="listbox"]',
    '[role="option"]',
    '[role="tab"]',
    '[role="tablist"]',
    '[role="treeitem"]',
    '[role="menu"]',
    '[role="menuitem"]',
    '[role="menuitemradio"]',
    '[role="menuitemcheckbox"]',
    '[role="dialog"]',
].join(',');
const MAX_CLICK_MOVEMENT_PX = 5;

const isSurfaceTarget = (card: HTMLElement, node: EventTarget | null): boolean => {
    if (!(node instanceof Node)) return false;
    // React portals bubble through their component ancestors, not their DOM ancestors.
    if (!card.contains(node)) return false;
    const target = node instanceof Element ? node : node.parentElement;
    if (target == null) return false;
    const control = target.closest(INTERACTIVE_TARGETS);
    // The outer mode radiogroup is outside this card and must not exclude its background.
    if (control != null && card.contains(control)) return false;
    return true;
};

/** Pointer enhancement only: native radios and switches remain the keyboard controls. */
export const isModeCardSurfaceClick = (event: CardClick): boolean => {
    if (
        event.defaultPrevented ||
        event.button !== 0 ||
        event.detail > 1 ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        !isSurfaceTarget(event.currentTarget, event.target)
    )
        return false;
    const card = event.currentTarget;
    return card.ownerDocument.getSelection()?.isCollapsed !== false;
};

export const useSurfaceActivation = (activate: () => void, isDisabled: boolean) => {
    const press = useRef<{ x: number; y: number; eligible: boolean } | null>(null);
    const onPointerDown = (event: SurfacePress) => {
        press.current = {
            x: event.clientX,
            y: event.clientY,
            eligible:
                !isDisabled &&
                !event.defaultPrevented &&
                event.button === 0 &&
                event.isPrimary !== false &&
                !event.altKey &&
                !event.ctrlKey &&
                !event.metaKey &&
                !event.shiftKey &&
                isSurfaceTarget(event.currentTarget, event.target),
        };
    };
    const onClick = (event: SurfaceClick) => {
        const start = press.current;
        press.current = null;
        // A control press or drag must not turn into a background click when released over padding.
        if (
            isDisabled ||
            (start != null &&
                (!start.eligible ||
                    Math.hypot(event.clientX - start.x, event.clientY - start.y) > MAX_CLICK_MOVEMENT_PX)) ||
            !isModeCardSurfaceClick(event)
        )
            return;
        activate();
    };
    return {
        onPointerDown,
        onClick,
        onPointerCancel: () => {
            if (press.current != null) press.current.eligible = false;
        },
    };
};
