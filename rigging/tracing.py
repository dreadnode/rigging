import logfire_api

Span = logfire_api.LogfireSpan

tracer = logfire_api.Logfire(otel_scope="rigging")
