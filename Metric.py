import config

#  return the config file es text

variables = {
    key: value
    for key, value in vars(config).items()
    if not key.startswith("__") and not callable(value)
    # and not isinstance(value, types.ModuleType)
}
