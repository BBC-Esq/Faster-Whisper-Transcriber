import os
import sys

def get_resource_path(relative_path: str) -> str:
    """Get the path to a resource file, handling both dev and installed scenarios."""
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if os.path.basename(script_dir) == "app" and os.path.exists(os.path.join(parent_dir, "compute_type.txt")):
            base_path = script_dir
        else:
            base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def get_install_dir() -> str:
    """Get the installation directory (parent of app directory when installed)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    if os.path.basename(script_dir) == "app" and os.path.exists(os.path.join(parent_dir, "compute_type.txt")):
        return parent_dir
    return script_dir


def is_gpu_install() -> bool:
    """Check if this is a GPU installation."""
    install_dir = get_install_dir()
    compute_type_file = os.path.join(install_dir, "compute_type.txt")

    if os.path.exists(compute_type_file):
        try:
            with open(compute_type_file, 'r') as f:
                return f.read().strip().lower() == "gpu"
        except Exception:
            pass

    return True
