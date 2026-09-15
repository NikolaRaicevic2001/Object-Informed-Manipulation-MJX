# Resolve the pixi env from this script's own location: $CONDA_PREFIX is
# unreliable here because an active conda/mambaforge base overrides it.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_here/../.pixi/envs/default/setup.bash"

# CycloneDDS config for THIS host: which NIC to bind (the lab ethernet, e.g.
# enp3s0 / enp114s0), so DDS never lands on a WiFi that happens to hold the
# default route. Per machine, so it lives in $HOME, not in the repo; copy
# ../config/cyclonedds.xml there and fill in the interface. Left unset when
# the file is missing: Cyclone then autodetermines the interface, which on
# a multi-homed host may be the wrong one, hence the warning.
if [ -f "$HOME/cyclonedds.xml" ]; then
    export CYCLONEDDS_URI="file://$HOME/cyclonedds.xml"
else
    echo "[dds] no ~/cyclonedds.xml on this host: CycloneDDS will pick a NIC itself" >&2
fi
