# offshore_energy_sim

Target package for the future integrated offshore energy simulation platform.

The current package layer wraps selected legacy RODM kernels with reusable
interfaces for configuration, structural assembly, hinges, reduction,
frequency-domain solving, response extraction, validation, strength helpers,
power helpers, and plotting.

Existing research scripts and notebooks remain the source of truth until each
workflow is migrated with reference-case checks. Package interfaces should keep
numerical results unchanged unless a task explicitly requests a numerical model
change and adds validation evidence.
