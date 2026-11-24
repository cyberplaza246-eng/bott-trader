# Serenity Peace at Home

A static website for **Serenity Peace at Home** — a transitional living facility for women and children.

**Tagline:** From Struggles to Strength · Healing. Hope. Home.

## Quick Start

No build step required. Open any HTML file directly in your browser, or use a local server for the best experience.

### Option 1: Open directly

Double-click `index.html` in File Explorer, or drag it into your browser.

### Option 2: Local server (recommended)

**Python 3:**
```bash
cd serenity-peace-at-home
python -m http.server 8080
```
Then visit [http://localhost:8080](http://localhost:8080)

**Node.js (npx):**
```bash
cd serenity-peace-at-home
npx serve .
```

**VS Code / Cursor:** Use the "Live Server" extension and open `index.html`.

## Project Structure

```
serenity-peace-at-home/
├── index.html          # Homepage
├── about.html          # Mission & about page
├── resources.html      # Community resource links
├── gallery.html        # Photo gallery (placeholder images)
├── contact.html        # Contact info & form
├── css/
│   └── styles.css      # All styles
├── js/
│   └── main.js         # Navigation, forms, carousel, gallery filters
├── assets/
│   └── images/
│       └── hero.png    # Main hero banner image
└── README.md
```

## Pages

| Page | Description |
|------|-------------|
| **Home** | Hero, about, services, journey path, testimonials, get-help form, community support |
| **About** | Expanded mission, vision, values, approach, team placeholder |
| **Resources** | Emergency shelters, food, mental health, jobs, housing, partners |
| **Gallery** | Filterable grid with CSS placeholder images |
| **Contact** | Contact details, form, map placeholder |

## Features

- Mobile-first responsive design
- Hamburger navigation on small screens
- Client-side form validation
- Testimonial carousel with autoplay
- Gallery category filters
- Smooth scroll for anchor links
- Semantic HTML & accessibility (skip link, ARIA labels, focus states)

## Customization Needed

The following placeholder content should be replaced with real information:

- **Phone:** (555) 123-4567
- **Email:** info@serenitypeaceathome.org
- **Address:** 123 Hope Lane, Community City, ST 12345
- **Office hours** on contact page
- **Social media links** (Facebook, Instagram, Twitter)
- **Google Maps embed** on contact page
- **Resource links** on resources page (marked as "Example")
- **Gallery photos** (currently CSS placeholders)
- **Team bios/photos** on about page
- **Testimonial names/details** (currently placeholder)
- **501(c)(3) status** — verify legal designation text

Forms currently show a success message on submit but do not send data anywhere. Connect to a backend or form service (e.g. Formspree, Netlify Forms) for production use.

## Design

- **Palette:** Cream/Ivory backgrounds, Soft Gold CTAs, Sage Green accents, Warm Beige sections, Light Brown headings
- **Fonts:** Great Vibes (brand), Lato (headings), Source Sans 3 (body) via Google Fonts

## License

© 2026 Serenity Peace at Home. All rights reserved.
