"""keras-remote CLI entry point."""

import click

from keras_remote.cli.commands.config import config
from keras_remote.cli.commands.down import down
from keras_remote.cli.commands.pool import pool
from keras_remote.cli.commands.status import status
from keras_remote.cli.commands.up import up


@click.group()
@click.version_option(package_name="keras-remote")
def cli():
  """keras-remote: Provision and manage GCP infrastructure for remote Keras
  execution."""


cli.add_command(up)
cli.add_command(down)
cli.add_command(status)
cli.add_command(config)
cli.add_command(pool)
