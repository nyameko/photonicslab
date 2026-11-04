# Integration with the research ecosystem

PhotonicsLab owns optical mathematical models, experimental configuration and physics validation. Quantum Workflows owns canonical scientific runners, notebook curriculum integration, reproducible run artifacts and Slurm/provider execution. Quantum Platform owns identity, programmes, access and resource/budget policy; infrastructure owns runtime placement and storage; ACP owns durable project/agent conversations and governed skill references.

The local `photonicslab` CLI emits its own `photonicslab.lesson-run/1` manifest. The existing example `examples/quantum_workflows_polarization.py` integrates with `quantum_workflows.artifacts.RunArtifacts` when that package is installed. **It is not yet a registered `qw` command or deployed portal resource offering.**

A proposed future production flow is:

```text
teacher/PI approves lesson -> Platform access
 -> notebook creates validated optical experiment
 -> Quantum Workflows runner with immutable environment
 -> Slurm/CPU/GPU if authorised
 -> scientific artifacts -> Platform result view
 -> ACP references experiment ID, not duplicated result binaries
```

Avoid a custom photonics cloud API until a concrete need justifies it; native Python notebooks are the primary interface. No H200, QPU or paid provider should be acquired implicitly by importing `photonicslab`.

See [adapter example](https://github.com/nyameko/photonicslab/blob/main/examples/quantum_workflows_polarization.py) and [architecture](https://github.com/nyameko/photonicslab/blob/main/docs/ARCHITECTURE.md).