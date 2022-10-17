# using Pkg	
# Pkg.activate("/scratch/users/mehta2/cyaxiverse/CYAxiverse")
using Revise
using Pluto
# cd("/scratch/users/mehta2/vacua_db")
Pluto.run(host="0.0.0.0", port=8996, require_secret_for_access=false)