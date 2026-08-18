---
name: Tropical Terminal
colors:
  surface: '#121414'
  surface-dim: '#121414'
  surface-bright: '#37393a'
  surface-container-lowest: '#0c0f0f'
  surface-container-low: '#1a1c1c'
  surface-container: '#1e2020'
  surface-container-high: '#282a2b'
  surface-container-highest: '#333535'
  on-surface: '#e2e2e2'
  on-surface-variant: '#bec9be'
  inverse-surface: '#e2e2e2'
  inverse-on-surface: '#2f3131'
  outline: '#899489'
  outline-variant: '#3f4940'
  surface-tint: '#83d99c'
  primary: '#83d99c'
  on-primary: '#00391b'
  primary-container: '#006837'
  on-primary-container: '#8ee4a6'
  inverse-primary: '#0b6d3b'
  secondary: '#ffffff'
  on-secondary: '#323200'
  secondary-container: '#eaea00'
  on-secondary-container: '#686800'
  tertiary: '#ffabf3'
  on-tertiary: '#5b005b'
  tertiary-container: '#a100a1'
  on-tertiary-container: '#ffbdf3'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#9ef6b6'
  primary-fixed-dim: '#83d99c'
  on-primary-fixed: '#00210e'
  on-primary-fixed-variant: '#00522a'
  secondary-fixed: '#eaea00'
  secondary-fixed-dim: '#cdcd00'
  on-secondary-fixed: '#1d1d00'
  on-secondary-fixed-variant: '#494900'
  tertiary-fixed: '#ffd7f5'
  tertiary-fixed-dim: '#ffabf3'
  on-tertiary-fixed: '#380038'
  on-tertiary-fixed-variant: '#810081'
  background: '#121414'
  on-background: '#e2e2e2'
  surface-variant: '#333535'
typography:
  headline-xl:
    fontFamily: Bodoni Moda
    fontSize: 120px
    fontWeight: '900'
    lineHeight: 110px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Bodoni Moda
    fontSize: 64px
    fontWeight: '800'
    lineHeight: 72px
  headline-lg-mobile:
    fontFamily: Bodoni Moda
    fontSize: 40px
    fontWeight: '800'
    lineHeight: 44px
  body-lg:
    fontFamily: JetBrains Mono
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: JetBrains Mono
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
spacing:
  unit: 8px
  container-max: 1280px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 48px
---

## Brand & Style

This design system draws inspiration from the intersection of Goa’s lush tropical landscapes and the raw, utilitarian aesthetic of hacker culture. It is designed for developers, creators, and digital nomads who seek a "work-from-paradise" lifestyle.

The style is a blend of **Brutalism** and **Vaporwave-inflected Tropicalism**. It utilizes high-contrast color pairings, bold condensed serif typography for impactful messaging, and monospaced fonts to reinforce the technical, "terminal" nature of the product. The UI should feel electric, energetic, and slightly rebellious—breaking away from sterile corporate SaaS norms in favor of a raw, expressive, and geographically rooted identity.

## Colors

The palette is built on a foundation of "Forest Deep" and "Solar Electric." 

- **Primary (#006837):** A deep, saturated forest green used as the primary canvas for all surfaces. It evokes the dense greenery of Goa.
- **Secondary (#FFFF00):** A vibrant, high-visibility yellow. This is the primary color for typography, primary buttons, and critical UI elements.
- **Tertiary (#FF00FF):** A hot magenta highlight. Use sparingly for decorative accents, special states, or to draw attention to specific "hacker" elements.
- **Neutral (#FFFFFF):** Pure white is used for secondary text, thin borders, and schematic illustrations to maintain legibility against the dark green background.

## Typography

The typography system relies on a stark contrast between high-fashion editorial serifs and technical monospaced fonts.

- **Headlines:** Use **Bodoni Moda** (Extra Bold/Black) for large display text. It should be tightly tracked and condensed to mimic the "HACKER HOUSE" logotype. This font brings an air of "luxury-meets-underground-zine."
- **Interface & Data:** Use **JetBrains Mono** for all functional UI elements, body text, and dates. This reinforces the "terminal" aesthetic and ensures clarity in dense information layouts.
- **Hierarchy:** Maintain a vertical rhythm by alternating between the two families. Critical information (dates, status) should always be monospaced.

## Layout & Spacing

The layout follows a **Fixed Grid** model for desktop and a **Fluid Fluid** model for mobile.

- **Grid:** A 12-column grid with generous margins (48px) to allow the deep green background to frame the content.
- **Rhythm:** Spacing is strictly based on an 8px base unit. Use larger gaps (64px+) between sections to create a "poster-like" feel rather than a traditional app layout.
- **Borders:** Use thin (1px) white or yellow lines to divide sections, mimicking technical blueprints or terminal window dividers.

## Elevation & Depth

This design system avoids traditional shadows and soft blurs. Depth is created through **Hard Layers** and **High-Contrast Outlines**.

- **Flat Hierarchy:** Elements exist on a single plane or are stacked with sharp 1px borders.
- **Inverted Surfaces:** To indicate elevation or focus, swap the background to yellow and text to green (Primary Button style).
- **Glassmorphism (Limited):** Use only for floating "terminal" overlays with a `backdrop-filter: blur(10px)` and a 1px white border at 20% opacity.

## Shapes

The shape language is strictly **Sharp (0px)**. 

Every UI element—buttons, input fields, cards, and containers—must have 90-degree corners. This reinforces the Brutalist and technical aesthetic. The only exceptions are specific decorative illustrations (like the sun or palm leaves) which provide a rhythmic contrast to the rigid UI.

## Components

- **Buttons:** Primary buttons are solid Yellow (#FFFF00) with Green (#006837) monospaced text. Secondary buttons use a 1px Yellow border with no fill. High-action buttons (Apply) can use a 1px Magenta (#FF00FF) shadow offset by 4px for a "glitch" effect.
- **Input Fields:** 1px White borders, sharp corners, and Yellow cursor/caret. Use monospaced labels positioned above the field.
- **Cards:** No shadows. Use a 1px border (White at 30% opacity) to define boundaries. Card headers should use Bodoni Moda.
- **Chips/Tags:** Small monospaced text inside a 1px border. Use Magenta for "Active" or "Hot" states.
- **Lists:** Bullet points are replaced by custom glyphs like `>` or `*` to mimic terminal lists.
- **Illustrations:** Use "Schematic Tropical" style—heavy outlines, flat colors, and minimal shading as seen in the reference beach scene.