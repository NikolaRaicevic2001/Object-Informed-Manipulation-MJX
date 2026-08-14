# Resolve the pixi env from this script's own location: $CONDA_PREFIX is
# unreliable here because an active conda/mambaforge base overrides it.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_here/../.pixi/envs/default/setup.bash"
