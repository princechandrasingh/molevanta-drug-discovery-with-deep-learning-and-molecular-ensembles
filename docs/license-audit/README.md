# Dependency notice audit

Generated 2026-09-23 from the actual isolated experiment environments.

- Pilot environment: **40 installed external distributions**, 76 shipped notice files.
- Reference environment: **69 installed distributions**, 85 shipped notice files. Five Sphinx extension wheels omitted their notices; the exact-version PyPI source archives supplied them. Archive SHA-256 values were verified against PyPI metadata before reading license text. No source archive code was executed.
- Every inventoried distribution now has at least one collected notice, with hashes verified locally. The inventories overlap; their package counts must not be added as unique software counts.
- This collects package-supplied notices and metadata; it is not a complete legal audit of every binary, bundled component, patent or data right. Metadata labels do not replace full license text. Missing/incomplete upstream notices can still exist.
- No installed dependencies are bundled into the project source archive. Keep the notices if distributing dependency code, and assess binary/environment distribution separately.

Machine-readable inventories: [pilot](pilot-inventory.json), [reference](reference-inventory.json), [supplemental exact-version source notices](supplemental-notices.json). The source URLs and license-file hashes are recorded there. Full notices are under `licenses/` and `source-notices/`.

Recreate the inventory from the project directory:

```bash
.venv/bin/python scripts/audit_licenses.py --environment pilot
.reference-venv/bin/python scripts/audit_licenses.py --environment reference
.venv/bin/python scripts/fetch_missing_notices.py
```

The fetch step reads public PyPI metadata and license text from source archives. It does not install those source packages.

## Environment inventory

Labels below are metadata descriptions, except the manually inspected descriptastorus notice. In particular, some packages include additional component licenses in the complete preserved notice.

| Environment | Distribution | Version | License metadata / notice | Notice source |
| --- | --- | --- | --- | --- |
| pilot | cloudpickle | 3.1.2 | BSD License | wheel |
| pilot | contourpy | 1.3.2 | BSD License | wheel |
| pilot | cycler | 0.12.1 | BSD License | wheel |
| pilot | et_xmlfile | 2.0.0 | MIT License | wheel |
| pilot | exceptiongroup | 1.3.1 | MIT License | wheel |
| pilot | filelock | 4.0.1 | MIT | wheel |
| pilot | fonttools | 4.65.0 | MIT | wheel |
| pilot | fsspec | 2026.9.0 | BSD-3-Clause | wheel |
| pilot | iniconfig | 2.3.0 | MIT | wheel |
| pilot | Jinja2 | 3.1.6 | BSD License | wheel |
| pilot | joblib | 1.6.0 | BSD-3-Clause | wheel |
| pilot | kiwisolver | 1.5.1 | BSD License | wheel |
| pilot | MarkupSafe | 3.0.3 | BSD-3-Clause | wheel |
| pilot | matplotlib | 3.10.9 | Python Software Foundation License | wheel |
| pilot | mpmath | 1.3.0 | BSD License | wheel |
| pilot | networkx | 3.4.2 | BSD License | wheel |
| pilot | numpy | 2.2.6 | BSD License | wheel |
| pilot | openpyxl | 3.1.5 | MIT License | wheel |
| pilot | packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | wheel |
| pilot | pandas | 2.3.3 | BSD License | wheel |
| pilot | pillow | 12.3.0 | MIT-CMU | wheel |
| pilot | pip | 23.0.1 | MIT License | wheel |
| pilot | pluggy | 1.6.0 | MIT License | wheel |
| pilot | Pygments | 2.21.0 | BSD-2-Clause | wheel |
| pilot | pyparsing | 3.3.3 | MIT | wheel |
| pilot | pytest | 8.4.2 | MIT License | wheel |
| pilot | python-dateutil | 2.9.0.post0 | BSD License; Apache Software License | wheel |
| pilot | pytz | 2026.3.post1 | MIT License | wheel |
| pilot | rdkit | 2025.9.6 | BSD-3-Clause | wheel |
| pilot | scikit-learn | 1.7.2 | BSD-3-Clause | wheel |
| pilot | scipy | 1.15.3 | BSD License | wheel |
| pilot | setuptools | 84.0.0 | MIT | wheel |
| pilot | six | 1.17.0 | MIT License | wheel |
| pilot | sympy | 1.14.0 | BSD License | wheel |
| pilot | threadpoolctl | 3.7.0 | BSD-3-Clause | wheel |
| pilot | tomli | 2.4.1 | MIT | wheel |
| pilot | torch | 2.10.0 | BSD-3-Clause | wheel |
| pilot | typing_extensions | 4.16.0 | PSF-2.0 | wheel |
| pilot | tzdata | 2026.4 | Apache-2.0 | wheel |
| pilot | wheel | 0.48.0 | MIT | wheel |
| reference | alabaster | 1.0.0 | BSD License | wheel |
| reference | babel | 2.18.0 | BSD License | wheel |
| reference | blinker | 1.9.0 | MIT License | wheel |
| reference | certifi | 2026.7.22 | Mozilla Public License 2.0 (MPL 2.0) | wheel |
| reference | charset-normalizer | 3.5.1 | MIT | wheel |
| reference | chemprop | 1.6.1 | MIT License | wheel |
| reference | click | 8.5.0 | BSD-3-Clause | wheel |
| reference | cloudpickle | 3.1.2 | BSD License | wheel |
| reference | contourpy | 1.3.2 | BSD License | wheel |
| reference | cycler | 0.12.1 | BSD License | wheel |
| reference | descriptastorus | 2.8.0 | BSD-3-Clause (shipped notice inspected) | wheel |
| reference | docstring_parser | 0.18.0 | MIT License | wheel |
| reference | docutils | 0.21.2 | Public Domain; Python Software Foundation License; BSD License; GNU General Public License (GPL) | wheel |
| reference | exceptiongroup | 1.3.1 | MIT License | wheel |
| reference | filelock | 4.0.1 | MIT | wheel |
| reference | Flask | 3.1.3 | BSD-3-Clause | wheel |
| reference | fonttools | 4.65.0 | MIT | wheel |
| reference | fsspec | 2026.9.0 | BSD-3-Clause | wheel |
| reference | hyperopt | 0.3.0 | BSD-3-Clause | wheel |
| reference | idna | 3.20 | BSD-3-Clause | wheel |
| reference | imagesize | 2.0.1 | MIT | wheel |
| reference | iniconfig | 2.3.0 | MIT | wheel |
| reference | itsdangerous | 2.2.0 | BSD License | wheel |
| reference | Jinja2 | 3.1.6 | BSD License | wheel |
| reference | joblib | 1.6.0 | BSD-3-Clause | wheel |
| reference | kiwisolver | 1.5.1 | BSD License | wheel |
| reference | MarkupSafe | 3.0.3 | BSD-3-Clause | wheel |
| reference | matplotlib | 3.10.9 | Python Software Foundation License | wheel |
| reference | mpmath | 1.3.0 | BSD License | wheel |
| reference | networkx | 3.4.2 | BSD License | wheel |
| reference | numpy | 1.26.4 | BSD License | wheel |
| reference | packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | wheel |
| reference | pandas | 2.3.3 | BSD License | wheel |
| reference | pandas-flavor | 0.8.1 | MIT License | wheel |
| reference | pillow | 12.3.0 | MIT-CMU | wheel |
| reference | pip | 23.0.1 | MIT License | wheel |
| reference | pluggy | 1.6.0 | MIT License | wheel |
| reference | protobuf | 7.36.2 | 3-Clause BSD License | wheel |
| reference | Pygments | 2.21.0 | BSD-2-Clause | wheel |
| reference | pyparsing | 3.3.3 | MIT | wheel |
| reference | pytest | 8.4.2 | MIT License | wheel |
| reference | python-dateutil | 2.9.0.post0 | BSD License; Apache Software License | wheel |
| reference | pytz | 2026.3.post1 | MIT License | wheel |
| reference | rdkit | 2025.9.6 | BSD-3-Clause | wheel |
| reference | requests | 2.34.2 | Apache Software License | wheel |
| reference | scikit-learn | 1.7.2 | BSD-3-Clause | wheel |
| reference | scipy | 1.12.0 | BSD License | wheel |
| reference | setuptools | 65.5.0 | MIT License | wheel |
| reference | six | 1.17.0 | MIT License | wheel |
| reference | snowballstemmer | 3.1.1 | BSD-3-Clause | wheel |
| reference | Sphinx | 8.1.3 | BSD License | wheel |
| reference | sphinxcontrib-applehelp | 2.0.0 | BSD License | PyPI sdist |
| reference | sphinxcontrib-devhelp | 2.0.0 | BSD License | PyPI sdist |
| reference | sphinxcontrib-htmlhelp | 2.1.0 | BSD License | PyPI sdist |
| reference | sphinxcontrib-jsmath | 1.0.1 | BSD License | wheel |
| reference | sphinxcontrib-qthelp | 2.0.0 | BSD License | PyPI sdist |
| reference | sphinxcontrib-serializinghtml | 2.0.0 | BSD License | PyPI sdist |
| reference | sympy | 1.14.0 | BSD License | wheel |
| reference | tensorboardX | 2.6.5 | MIT | wheel |
| reference | threadpoolctl | 3.7.0 | BSD-3-Clause | wheel |
| reference | tomli | 2.4.1 | MIT | wheel |
| reference | torch | 2.2.2 | BSD License | wheel |
| reference | tqdm | 4.70.1 | MPL-2.0 AND MIT | wheel |
| reference | typed-argument-parser | 1.12.0 | MIT | wheel |
| reference | typing_extensions | 4.16.0 | PSF-2.0 | wheel |
| reference | tzdata | 2026.4 | Apache-2.0 | wheel |
| reference | urllib3 | 2.8.0 | MIT | wheel |
| reference | Werkzeug | 3.1.8 | BSD-3-Clause | wheel |
| reference | xarray | 2025.6.1 | Apache-2.0 | wheel |
