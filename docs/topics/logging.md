# Logging

Rigging uses [loguru](https://loguru.readthedocs.io/) for it's logging. By default it disables it's logger allowing users to choose when/how to gather messages.

If you want to let rigging messages flow into loguru, you should enable it:

```py
from loguru import logger

logger.enable('rigging')
```

If you want to have some sane default handlers with dual console & file logging,
you can use the [rigging.logging.configure_logging][] function to configure loguru. 

```py
from rigging.logging import configure_logging

configure_logging(
    'info',      # stderr level
    'out.log',   # log file (optional)
    'trace'      # log file level
)
```
*(This will remove existing handlers, so you might prefer to configure them yourself)*
