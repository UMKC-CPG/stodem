# CMake file for STODEM set of scripts.

# Document the current location
message("At the set of scripts.")

set (SCRIPTS
stodem.py
)

install (PROGRAMS ${SCRIPTS} DESTINATION bin)
