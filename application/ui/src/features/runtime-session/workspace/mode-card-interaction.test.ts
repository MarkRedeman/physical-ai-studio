import { describe, expect, it, rs } from '@rstest/core';

import { renderHook } from '../../../test-utils/render';
import { isModeCardSurfaceClick, useSurfaceActivation } from './mode-card-interaction';

const nodes: HTMLElement[] = [];
const makeCard = (html = '<div><h3>Recording</h3><p>Dataset name</p></div>') => {
    const card = document.createElement('section');
    card.innerHTML = html;
    document.body.append(card);
    nodes.push(card);
    return card;
};
const click = (
    card: HTMLElement,
    target: EventTarget = card,
    overrides: Partial<Parameters<typeof isModeCardSurfaceClick>[0]> = {}
) =>
    isModeCardSurfaceClick({
        currentTarget: card,
        target,
        defaultPrevented: false,
        button: 0,
        detail: 1,
        altKey: false,
        ctrlKey: false,
        metaKey: false,
        shiftKey: false,
        ...overrides,
    });

afterEach(() => {
    document.getSelection()?.removeAllRanges();
    nodes.splice(0).forEach((node) => node.remove());
});

describe('mode card surface clicks', () => {
    it('accepts the card background and non-interactive header space', () => {
        const card = makeCard();
        expect(click(card)).toBe(true);
        expect(click(card, card.querySelector('h3') ?? card)).toBe(true);
    });

    it.each([
        'button',
        'input',
        'textarea',
        'select',
        'label',
        'summary',
        'a href="/models"',
        'div role="button"',
        'div role="slider"',
        'div role="radiogroup"',
        'div role="combobox"',
        'div role="menuitem"',
        'div tabindex="0"',
        'div contenteditable="true"',
    ])('excludes interactive targets: %s', (openingTag) => {
        const tag = openingTag.split(' ')[0];
        const card = makeCard(`<${openingTag} data-target>Content</${tag}>`);
        const target = card.querySelector('[data-target]');
        expect(target).not.toBeNull();
        expect(click(card, target ?? card)).toBe(false);
    });

    it('excludes nested icons and text inside disabled buttons', () => {
        const card = makeCard('<button disabled><span>Choose model</span><svg><path d="M0 0" /></svg></button>');
        expect(click(card, card.querySelector('span') ?? card)).toBe(false);
        expect(click(card, card.querySelector('path') ?? card)).toBe(false);
    });

    it('excludes slider tracks and padding inside an explicit control area', () => {
        const card = makeCard('<div data-mode-interactive><div data-track>Slider track</div></div>');
        expect(click(card, card.querySelector('[data-track]') ?? card)).toBe(false);
    });

    it('does not confuse the surrounding mode radiogroup with an inner control', () => {
        const card = makeCard();
        const outer = document.createElement('div');
        outer.setAttribute('role', 'radiogroup');
        document.body.append(outer);
        nodes.push(outer);
        outer.append(card);
        expect(click(card)).toBe(true);
    });

    it('rejects events from portaled popover content outside the card', () => {
        const card = makeCard();
        const popup = document.createElement('div');
        document.body.append(popup);
        nodes.push(popup);
        expect(click(card, popup)).toBe(false);
    });

    it('does not switch modes while selecting text', () => {
        const card = makeCard();
        const text = card.querySelector('p') ?? card;
        const range = document.createRange();
        range.selectNodeContents(text);
        document.getSelection()?.addRange(range);
        expect(click(card, text)).toBe(false);
    });

    it.each([{ defaultPrevented: true }, { button: 2 }, { detail: 2 }, { ctrlKey: true }, { shiftKey: true }])(
        'ignores handled, secondary, or modified clicks: %j',
        (overrides) => {
            expect(click(makeCard(), undefined, overrides)).toBe(false);
        }
    );
});

describe('surface gesture boundaries', () => {
    const press = (card: HTMLElement, target: EventTarget, x = 10, y = 10) => ({
        currentTarget: card,
        target,
        clientX: x,
        clientY: y,
        defaultPrevented: false,
        button: 0,
        isPrimary: true,
        altKey: false,
        ctrlKey: false,
        metaKey: false,
        shiftKey: false,
    });
    const release = (card: HTMLElement, x = 10, y = 10) => ({ ...press(card, card, x, y), detail: 1 });

    it('never activates the panel for a press that started on a control', () => {
        const card = makeCard('<input>');
        const activate = rs.fn();
        const { result } = renderHook(() => useSurfaceActivation(activate, false));
        result.current.onPointerDown(press(card, card.querySelector('input') ?? card));
        result.current.onClick(release(card));
        expect(activate).not.toHaveBeenCalled();
    });

    it('rejects a drag but accepts a normal background click', () => {
        const card = makeCard();
        const activate = rs.fn();
        const { result } = renderHook(() => useSurfaceActivation(activate, false));
        result.current.onPointerDown(press(card, card));
        result.current.onClick(release(card, 30, 30));
        expect(activate).not.toHaveBeenCalled();
        result.current.onPointerDown(press(card, card));
        result.current.onClick(release(card, 11, 11));
        expect(activate).toHaveBeenCalledTimes(1);
    });

    it('does not activate after pointer cancellation', () => {
        const card = makeCard();
        const activate = rs.fn();
        const { result } = renderHook(() => useSurfaceActivation(activate, false));
        result.current.onPointerDown(press(card, card));
        result.current.onPointerCancel();
        result.current.onClick(release(card));
        expect(activate).not.toHaveBeenCalled();
    });
});
