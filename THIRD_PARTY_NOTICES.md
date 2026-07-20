# Third-party software notices

RHO Planner packages third-party open-source Python and native libraries. Their licenses remain applicable to the bundled copies.

Important components include:

- Qt for Python / PySide6 and Qt libraries — LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only
- PyInstaller — GPL-2.0-or-later with a special exception for distributing bundled applications
- Astropy and astroplan — BSD-3-Clause
- Astroquery — BSD-3-Clause
- NumPy — BSD-3-Clause
- SciPy — BSD-3-Clause
- pandas — BSD-3-Clause
- Matplotlib — PSF-based license
- Pillow — HPND
- Requests — Apache-2.0
- openpyxl — MIT
- certifi — MPL-2.0

Before publishing binaries, retain the repository's own license and review the exact license files included with the versions recorded in each release's `DEPENDENCIES-*.txt` manifest. PySide6 is dynamically bundled from its official wheels; distributors should preserve applicable Qt/PySide notices and comply with the LGPL terms.

This notice is informational and is not legal advice.
