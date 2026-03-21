import subprocess
import sys
from typing import Any, Dict


def background_process_kwargs(new_process_group: bool = False) -> Dict[str, Any]:
    """Return platform-specific Popen kwargs for hidden background processes."""
    kwargs: Dict[str, Any] = {}

    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if new_process_group:
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if creationflags:
            kwargs["creationflags"] = creationflags

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["startupinfo"] = startupinfo
    elif new_process_group:
        kwargs["start_new_session"] = True

    return kwargs
