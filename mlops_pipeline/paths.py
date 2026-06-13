import os

_PROJECT_ROOT = None


def get_project_root():
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    return _PROJECT_ROOT


def get_path(*parts):
    return os.path.join(get_project_root(), *parts)


def load_config():
    import yaml
    config_path = get_path("config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_path(config_relative_path):
    if os.path.isabs(config_relative_path):
        return os.path.normpath(config_relative_path)
    return os.path.normpath(get_path(config_relative_path))
