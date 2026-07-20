import logging

from recomendacion.config.core import PACKAGE_ROOT, config

# Logger del paquete con NullHandler para no restringir las aplicaciones
# que consuman el paquete. Ver:
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
