# CMake file for STODEM set of scripts.

# Document the current location
message("At the set of scripts.")

set (SCRIPTS
citizen.py
diagnostics.py
gaussian.py
government.py
output.py
patch.py
politician.py
random_state.py
settings.py
sim_control.py
stodem.py
world.py
zone.py
)

install (PROGRAMS ${SCRIPTS} DESTINATION bin)
