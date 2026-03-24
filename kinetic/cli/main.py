"""kinetic CLI entry point."""

import click

from kinetic.cli.commands.config import config
from kinetic.cli.commands.down import down
from kinetic.cli.commands.pool import pool
from kinetic.cli.commands.status import status
from kinetic.cli.commands.up import up


@click.group()
@click.version_option(package_name="kinetic")
def cli():
  """kinetic: Provision and manage GCP infrastructure for remote Keras
  execution."""


cli.add_command(up)
cli.add_command(down)
cli.add_command(status)
cli.add_command(config)
cli.add_command(pool)
