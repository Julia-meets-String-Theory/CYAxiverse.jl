using Pkg
Pkg.instantiate()
using Revise
using Pluto
# Set CYAXIVERSE_DATA_DIR to select a database outside the container default.
Pluto.run(host="0.0.0.0", port=8994, require_secret_for_access=false)
