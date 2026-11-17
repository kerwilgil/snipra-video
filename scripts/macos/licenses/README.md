# Licencias y avisos de Snipra 1.0.0 (macOS arm64)

Esta carpeta acompaña a `Snipra.app` (Contents/Resources/licenses). Se genera en cada build
con `scripts/macos/license_inventory.py` a partir del propio bundle y de las fórmulas Homebrew.

| Fichero | Contenido |
|---|---|
| `THIRD_PARTY_NOTICES.md` | Inventario legible: componente, versión, licencia y origen de cada pieza. |
| `SOURCES.md` | Oferta de código fuente correspondiente: URL exacta y sha256 de cada componente copyleft. |
| `inventory.json` | El mismo inventario en formato máquina, con el campo `review` de revisión legal. |
| `formulae/` | Fórmulas Homebrew (`.rb`) usadas: describen opciones de compilación y parches. |
| `components/` | Ficheros de licencia y aviso que traen las propias fuentes de cada componente. |
| `texts/` | Textos canónicos (SPDX) de las licencias presentes. |
| `LICENSE-Snipra` | Licencia MIT de Snipra. |
| `homebrew-bottles.json` | Bottles de Homebrew usados (formula, versión, sha256 y URL). |

Obtención del código fuente completo: `scripts/macos/fetch-corresponding-source.py --inventory inventory.json --out DIR`.

Estado: pendiente de revisión jurídica (LEGAL_REVIEW_REQUIRED). La distribuci?n de binarios requiere una revisi?n jur?dica espec?fica.
