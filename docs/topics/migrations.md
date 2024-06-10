# Migrations

As we continue to develop and improve Rigging, we may introduce changes that
break backwards compatibility and/or signficantly change mechanics of the library.

In general we try to follow best practices for semantic versioning:

- **Major version**: Significant and breaking changes (e.g. v1.X to v2.X)
- **Minor version**: New features or improvements (e.g. v1.0 to v1.1)
- **Patch version**: Bug fixes or minor improvements (e.g. v1.0.0 to v1.0.1)

## Migrating from v1.x to v2.x

### Rigging is now exclusivley async

Maintaining dual interface support was complex and error-prone, and we always
tried to implement the more performant code in the async interface.

Ideally we could have maintained synchronous "gates" which managed asyncio loops for the user, but this
is has caveats in notebook/jupyter environments. Ultimately we've decided to migrate
exclusively to async to simplify the codebase and improve performance.

- There are no longer any `a`-prefixed functions. Functions like [`run`][rigging.chat.ChatPipeline.run] and
  [`generate_messages`][rigging.generator.Generator.generate_messages] are now coroutines that need to be awaited.
- [`map`][rigging.chat.ChatPipeline.map] and [`then`][rigging.chat.ChatPipeline.then] callbacks are now expected to be coroutines.

Adapting these changes should be relatively straightforward. `await` can be used directly in Jupyter nodebooks
by default. Wrapping any entrypoint with `asyncio.run(...)` is a simple way to manage an event loop. If you're
in a more unique scenario, check out the [greenback](https://github.com/oremanj/greenback) to allow stepping in/out
of async code in a larger system.

We also provide a helper [`await_`][rigging.util.await_] function which can be used in place
of standard `await` in synchronous code. Underneath rigging will manage an event loop for you
in a separate thread and pass coroutines into it for resolution.


=== "rg.await_()"

    ```py
    import rigging as rg

    def main():
        generator = rg.get_generator(...)
        pipeline = generator.chat(...)

        chat = rg.await_(pipeline.run()) # (1)!
    
    if __name__ == "__main__":
        main()
    ```

    1. You can pass a single coroutine or a positional list of coroutines to [`await_`][rigging.util.await_].


=== "asyncio.run()"

    ```py
    import asyncio
    import rigging as rg

    async def main():
        generator = rg.get_generator(...)
        pipeline = generatore.chat(...)

        chat = await pipeline.run()

    if __name__ == "__main__":
        asyncio.run(main())
    ```


### "Pending" -> "Pipeline"

Language around chat pipelines and completions was confusing, and didn't accurately
communicate the power of the pipeline system. We've renamed `PendingChat` to `ChatPipeline` and
`PendingCompletion` to `CompletionPipeline`.

This shouldn't affect most users unless you were manually accessing these classes. You'll see us
replace the frequently use of `pending` variables with `pipeline` in our code.

### `on_failed` replaces `skip_failed`/`include_failed`

Pipelines now provide better clarity for catching errors and translating
them into failed outputs. We've replaced the `skip_failed` and `include_failed`
arguments for a general string literal `on_failed` mapped to [`FailMode`][rigging.chat.FailMode].

This should help us clarity behaviors and expand them in the future without causing
argument bloat.