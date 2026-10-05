# Changelog

Notable changes and upgrade actions are recorded here.

## [Unreleased]

### Added

- Configurable `high_pass_hz` for `time_domain_filter_diver_style` and
  `notch_freqs` for `time_domain_filter`.

### Changed

- Bundled preprocessing YAMLs explicitly declare high-pass cutoffs and notch
  frequencies, preserving their previous numerical behavior.
- High-pass settings and standard-filter notch lists reject invalid types and
  non-finite values; high-pass cutoffs must be nonnegative and below Nyquist.

### Fixed

- Standard-filter notch frequencies and DIVER high-pass cutoffs now honor the
  config; previously these fields were ignored in favor of hard-coded values.
- Preprocessing caches are invalidated so results computed before those fields
  took effect cannot be reused under the same config.

### Upgrade notes

- Default behavior is preserved: omitted standard-filter `high_pass_hz` means
  `0.0`; omitted or `null` DIVER `high_pass_hz` means `0.5`. Omitted notch lists
  retain each filter's historical frequencies. Default configs need no edits.
- Check custom configs that already set standard-filter `notch_freqs` or DIVER
  `high_pass_hz`: those values now affect processing. Use a new output root if
  effective settings change; existing result JSONs are skipped by filename.
- Rebuild train-source and preprocessed-split caches with `read_write` or
  `refresh` before using `read_only` mode.

## [0.1.0] - 2026-09-20

### Added

- Initial versioned release of the public iMINDBench evaluation package.
- Evaluation workflows for NeuroprobeV2, Bang! You're Dead, and PIPPI, with
  bundled preprocessing configs, baseline and pretrained-model integrations,
  and result JSON export.
- Dataset launch scripts for within-session evaluation and supported transfer
  and sample-efficiency experiments.

### Release notes

- Establishes the existing `main` implementation as the versioned baseline;
  no preprocessing, training, or evaluation behavior changes in this release.
- CPU evaluation supports Python 3.10–3.13; GPU models on newer Python versions
  are not yet validated. See the README for dependencies and checkpoint setup.
