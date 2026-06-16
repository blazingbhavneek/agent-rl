import inspect


class State:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(State, cls).__new__(cls)
            # Initialize with an empty storage if first time
            print(f"STATE INTIALIZED")
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, initial_vars: dict[str, any] = None):
        # Ensure __init__ only runs once even if called multiple times
        if self._initialized:
            if initial_vars:
                print(
                    "Warning(soft): Don't set the value in the constructor as the object is already intialized. Use set instead"
                )
            return

        if initial_vars:
            for name, value in initial_vars.items():
                setattr(self, name, value)

        self._initialized = True

    def set(self, name: str, value: any):
        """Sets a variable. Warns if it already exists."""
        if hasattr(self, name):
            print(
                f"⚠️  Warning: '{name}' is already defined in STATE. Overwriting...",
                f"Last overwritten in: {inspect.stack()[-1].filename}: line {inspect.stack()[-1].lineno}",
            )  # print the last stack

        setattr(self, name, value)

    def get(self, name: str, default: any = None):
        """Gets a variable. Throws error if missing and no default provided."""
        if not hasattr(self, name):
            return default

        return getattr(self, name)

    def reset(self):
        """Resets the state without breaking the class mechanics."""
        # List of internal attributes we MUST keep
        internal_attrs = {"_initialized", "TOOLS"}

        for attr in list(self.__dict__.keys()):
            if attr not in internal_attrs:
                delattr(self, attr)

        # Keep it initialized so the next __init__ call doesn't overwrite everything
        self._initialized = True
        print(f"STATE RESET (Data cleared, mechanics preserved)")


# Instantiate the singleton for project-wide use
# STATE = State({"app_name": "MyProject", "version": 1.0})
