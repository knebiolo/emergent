# -*- coding: utf-8 -*-

try:
	from emergent.sockeye_dynamic_environment import *
except Exception:
	# Allow importing this package in test environments where optional
	# legacy modules may not be available. Tests import submodules
	# directly by path to avoid side-effects.
	pass

#from emergent.ship import *
