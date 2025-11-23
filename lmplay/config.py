"""Configuration file loading and saving for LMPlay.

Handles YAML config files with three sections:
- construction_args: Parameters for constructing new models/optimizers/schedulers
- state_args_overrides: Optional modifications to saved runtime state
- run_args: Command-line argument defaults (can be overridden by CLI)
"""

import os
import yaml
from typing import Dict, Any, Tuple


class ConfigError(Exception):
  """Raised when config file loading or parsing fails."""
  pass


class _DefaultValue:
  """Marker class to detect if a command-line argument was explicitly provided.

  Used as default value in argparse to distinguish between:
  - User explicitly provided a value (replaced this marker)
  - User didn't provide it (still has this marker)

  This allows config file to set defaults only for args the user didn't specify.
  """
  def __init__(self, value):
    """Store the default value."""
    self.value = value


def apply_config_defaults(args, run_args_from_config):
  """Apply config run_args defaults only to args the user didn't explicitly set.

  Args:
    args: Parsed argparse Namespace object
    run_args_from_config: Dict of run_args from config file
  """
  for attr_name in vars(args):
    current_value = getattr(args, attr_name)

    # If this arg was not provided by user (still has _DefaultValue marker)
    if isinstance(current_value, _DefaultValue):
      # Try to override from config, else use the wrapped default value
      if attr_name in run_args_from_config:
        setattr(args, attr_name, run_args_from_config[attr_name])
      else:
        # Keep the default value from the marker
        setattr(args, attr_name, current_value.value)


def load_config(args, location: str = None, default: str = 'lmpdefaults.yaml') -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Load YAML config file with fallback to default, and apply run_args to args.

  If location is specified, it must exist (error if not).
  If location is None, tries to load default file if it exists, otherwise returns empty dicts.

  Modifies args in place by applying run_args defaults from config to any _DefaultValue markers.

  Args:
    args: Parsed argparse Namespace object to apply run_args to
    location: Path to config file. If None, tries to load default.
    default: Path to default config file to try if location is None

  Returns:
    tuple: (construction_args, state_args_overrides)

  Raises:
    ConfigError: If location is specified but doesn't exist or is invalid
  """
  construction_args, state_args_overrides, run_args = ({}, {}, {})

  if location:
    # Explicit location specified - must succeed
    construction_args, state_args_overrides, run_args = _load_config_file(location)
  else:
    # Try default, but don't error if it doesn't exist
    if os.path.exists(default):
      try:
        construction_args, state_args_overrides, run_args = _load_config_file(default)
      except ConfigError:
        pass

  # Apply run_args to args object in place
  apply_config_defaults(args, run_args)

  return construction_args, state_args_overrides


def _load_config_file(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
  """Internal function to load YAML config file.

  Args:
    config_path: Path to YAML config file

  Returns:
    tuple: (construction_args, state_args_overrides, run_args)

  Raises:
    ConfigError: If file doesn't exist or is invalid YAML
  """
  if not os.path.exists(config_path):
    raise ConfigError(f"Config file not found: {config_path}")

  try:
    with open(config_path, 'r') as f:
      config = yaml.safe_load(f) or {}
  except yaml.YAMLError as e:
    raise ConfigError(f"Invalid YAML in {config_path}: {e}")

  if not isinstance(config, dict):
    raise ConfigError(f"Config must be a dict, got {type(config)}")

  construction_args = config.get('construction_args', {})
  state_args_overrides = config.get('state_args_overrides', {})
  run_args = config.get('run_args', {})

  return construction_args, state_args_overrides, run_args


def save_config(config_path: str, construction_args: Dict[str, Any],
                state_args: Dict[str, Any], comment_state_args: bool = True) -> None:
  """Save config to YAML file.

  Args:
    config_path: Path where to write config file
    construction_args: Construction parameters by section
    state_args: Current state parameters by section
    comment_state_args: If True, write state_args as commented examples. If False, write uncommented.
  """
  os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)

  with open(config_path, 'w') as f:
    # Write construction_args section
    yaml.dump({'construction_args': construction_args}, f,
              default_flow_style=False, sort_keys=False)

    # Write state_args section (commented or uncommented)
    if state_args:
      f.write('\n')
      if comment_state_args:
        f.write('# state_args_overrides: Uncomment and modify to override saved state\n')
        f.write('# ')
        state_args_yaml = yaml.dump({'state_args_overrides': state_args},
                                    default_flow_style=False, sort_keys=False)
        f.write(state_args_yaml.replace('\n', '\n# '))
      else:
        yaml.dump({'state_args_overrides': state_args}, f,
                  default_flow_style=False, sort_keys=False)