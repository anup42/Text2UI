# Text2UI Icon Set

This directory hosts a lightweight set of assistant-friendly SVG icons that can be embedded directly inside generated HTML. Each icon is sized on a 24×24 grid and uses stroked outlines so it can adapt to the existing `agent2.css` button and layout utilities without additional styling.

## Base URL

When the dataset needs to reference an icon, use the raw GitHub URL for this repository:

```
https://raw.githubusercontent.com/AnySphereAI/Text2UI/main/data/icons/<icon-name>.svg
```

Replace `<icon-name>` with one of the filenames listed in [`manifest.json`](./manifest.json). The files are ASCII-only and minified for compatibility with the prompt rules.

## Usage guidance

- Use icons sparingly to reinforce meaning (e.g., payment, alerts, navigation).
- Inline icons should appear before their associated text so screen reader order remains logical.
- For decorative-only icons, set `alt=""` and `aria-hidden="true"` on the `<img>` element. Provide descriptive `alt` text when the icon conveys essential information.
- Keep icons inside existing utility structures (buttons, list items, callouts) without adding new classes or inline styles. Width and height attributes (e.g., `width="20" height="20"`) are sufficient.

Refer to the prompt updates in `data/prompts/gemini_dataset_prompt_actions_icons.txt` for the exact authoring rules.
