import typing as t

import logfire_api

Span = logfire_api.LogfireSpan


class Tracer(logfire_api.Logfire):
    def span(
        self,
        msg_template: str,
        /,
        *,
        _tags: t.Sequence[str] | None = None,
        _span_name: str | None = None,
        _level: t.Any | None = None,
        _links: t.Any = (),
        **attributes: t.Any,
    ) -> logfire_api.LogfireSpan:
        # Pass msg_template as the span name
        # to avoid weird fstring behaviors
        return super().span(
            msg_template,
            _tags=_tags,
            _span_name=msg_template,
            _level=_level,
            _links=_links,
            **attributes,
        )


tracer = Tracer(otel_scope="rigging")
