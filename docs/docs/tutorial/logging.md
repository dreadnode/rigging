### Logging

By default rigging disables it's logger with loguru. To enable it run:

```python
from loguru import logger

logger.enable('rigging')
```

To configure loguru terminal + file logging format overrides:

```python
from rigging.logging import configure_logging

configure_logging(
    'info',      # stderr level
    'out.log',   # log file (optional)
    'trace'      # log file level
)
```
*(This will remove existing handlers, so you might prefer to configure them yourself)*
