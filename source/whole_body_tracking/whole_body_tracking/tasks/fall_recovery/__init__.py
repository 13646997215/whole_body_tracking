"""Fall recovery / stand-up tasks for custom humanoid."""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
