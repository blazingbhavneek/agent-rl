import time
from functools import wraps

from state.state import State


def time_it(message: str | None = None):
    # This level receives the function
    def decorator(func):
        @wraps(func)  # Best practice: keeps your function's metadata intact
        def wrapper(*args, **kwargs):
            project_name = State().get("PROJECT_NAME")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            display_msg = f" ({message})" if message else ""
            print(
                f"Function {func.__name__} {'' if not display_msg else f'Message: {display_msg}'} (Project: {project_name}) took: {end_time-start_time:.4f}s"
            )

            return result

        return wrapper

    return decorator
