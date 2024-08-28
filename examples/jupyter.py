import asyncio
import contextlib
import datetime
import io
import json
import pathlib
import random
import time
import typing as t
import uuid
from dataclasses import dataclass

import aiodocker
import aiodocker.utils
import click
import httpx
import websockets
from loguru import logger
from typing_extensions import Self
from websockets.client import WebSocketClientProtocol

import rigging as rg
from rigging import logging

# Much of the Jupyter code was stolen from https://github.com/microsoft/autogen
# ... so thank you! 🙏


@dataclass
class JupyterConnectionInfo:
    host: str
    use_https: bool
    port: int | None = None
    token: str | None = None


@dataclass
class JupyterExecutionResult:
    @dataclass
    class DataItem:
        mime_type: str
        data: str

    status: t.Literal["ok", "error"]
    output: str
    data_items: list[DataItem]


class JupyterKernelClient:
    def __init__(self, websocket: WebSocketClientProtocol):
        self._session_id: str = uuid.uuid4().hex
        self._websocket: WebSocketClientProtocol = websocket

    async def close(self) -> None:
        await self._websocket.close()

    async def _send_message(self, *, content: dict[str, t.Any], channel: str, message_type: str) -> str:
        timestamp = datetime.datetime.now().isoformat()
        message_id = uuid.uuid4().hex
        message = {
            "header": {
                "username": "",
                "version": "5.0",
                "session": self._session_id,
                "msg_id": message_id,
                "msg_type": message_type,
                "date": timestamp,
            },
            "parent_header": {},
            "channel": channel,
            "content": content,
            "metadata": {},
            "buffers": {},
        }
        await self._websocket.send(json.dumps(message))
        return message_id

    async def _receive_message(self, timeout_seconds: float | None) -> dict[str, t.Any] | None:
        try:
            data = await asyncio.wait_for(self._websocket.recv(), timeout=timeout_seconds)
            return t.cast(dict[str, t.Any], json.loads(data))
        except asyncio.TimeoutError:
            return None

    async def wait_for_ready(self, timeout_seconds: float | None = None) -> bool:
        message_id = await self._send_message(content={}, channel="shell", message_type="kernel_info_request")
        while True:
            message = await self._receive_message(timeout_seconds)
            if message is None:
                break

            if (
                message.get("parent_header", {}).get("msg_id") == message_id
                and message["msg_type"] == "kernel_info_reply"
            ):
                return True

        logger.debug(f"Kernel did not become ready within {timeout_seconds} seconds. [{self._session_id}]")

        return False

    async def execute(self, code: str, timeout_seconds: float | None = None) -> JupyterExecutionResult:
        message_id = await self._send_message(
            content={
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            channel="shell",
            message_type="execute_request",
        )

        text_output = []
        data_output = []
        while True:
            message = await self._receive_message(timeout_seconds)
            if message is None:
                logger.debug(f"Timeout waiting for output from code block. [{self._session_id}]")
                return JupyterExecutionResult(
                    status="error", output="ERROR: Timeout waiting for output from code block.", data_items=[]
                )

            # Ignore messages that are not for this execution.
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue

            msg_type = message["msg_type"]
            content = message["content"]

            if msg_type in ["execute_result", "display_data"]:
                for data_type, data in content["data"].items():
                    if data_type == "text/plain":
                        text_output.append(data)
                    else:
                        data_output.append(JupyterExecutionResult.DataItem(mime_type=data_type, data=data))

            elif msg_type == "stream":
                text_output.append(content["text"])

            elif msg_type == "error":
                logger.debug(f"Error executing code block: {content['ename']} [{self._session_id}]")
                return JupyterExecutionResult(
                    status="error",
                    output=f"ERROR: {content['ename']}: {content['evalue']}\n{content['traceback']}",
                    data_items=[],
                )

            if msg_type == "status" and content["execution_state"] == "idle":
                break

        return JupyterExecutionResult(
            status="ok", output="\n".join([str(output) for output in text_output]), data_items=data_output
        )


class JupyterGatewayClient:
    def __init__(self, connection_info: JupyterConnectionInfo):
        self._connection_info = connection_info
        self._client = httpx.AsyncClient()

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self._connection_info.token}"} if self._connection_info.token else {}

    @property
    def api_base_url(self) -> str:
        protocol = "https" if self._connection_info.use_https else "http"
        port = f":{self._connection_info.port}" if self._connection_info.port else ""
        return f"{protocol}://{self._connection_info.host}{port}"

    @property
    def ws_base_url(self) -> str:
        protocol = "wss" if self._connection_info.use_https else "ws"
        port = f":{self._connection_info.port}" if self._connection_info.port else ""
        return f"{protocol}://{self._connection_info.host}{port}"

    async def close(self) -> None:
        await self._client.aclose()

    async def is_ready(self) -> bool:
        try:
            response = await self._client.get(f"{self.api_base_url}/api", headers=self.headers)
            return response.status_code == 200
        except Exception:
            return False

    async def list_kernel_specs(self) -> dict[str, dict[str, str]]:
        response = await self._client.get(f"{self.api_base_url}/api/kernelspecs", headers=self.headers)
        response.raise_for_status()
        return t.cast(dict[str, dict[str, str]], response.json())

    async def list_kernels(self) -> list[dict[str, str]]:
        response = await self._client.get(f"{self.api_base_url}/api/kernels", headers=self.headers)
        response.raise_for_status()
        return t.cast(list[dict[str, str]], response.json())

    async def start_kernel(self, kernel_spec_name: str) -> str:
        response = await self._client.post(
            f"{self.api_base_url}/api/kernels",
            headers=self.headers,
            json={"name": kernel_spec_name},
        )
        if response.status_code != 201:
            raise RuntimeError(f"Failed to start kernel: {response.text}")
        return t.cast(str, response.json()["id"])

    async def delete_kernel(self, kernel_id: str) -> None:
        response = await self._client.delete(f"{self.api_base_url}/api/kernels/{kernel_id}", headers=self.headers)
        response.raise_for_status()

    async def restart_kernel(self, kernel_id: str) -> None:
        response = await self._client.post(f"{self.api_base_url}/api/kernels/{kernel_id}/restart", headers=self.headers)
        response.raise_for_status()

    async def get_kernel_client(self, kernel_id: str) -> JupyterKernelClient:
        ws_url = f"{self.ws_base_url}/api/kernels/{kernel_id}/channels"
        websocket = await websockets.connect(ws_url, extra_headers=self.headers)
        return JupyterKernelClient(websocket)

    @contextlib.asynccontextmanager
    async def create_kernel(self, kernel_spec_name: str) -> t.AsyncGenerator[JupyterKernelClient, None]:
        kernel_id = await self.start_kernel(kernel_spec_name)
        logger.debug(f"Kernel {kernel_id} started.")

        try:
            kernel_client = await self.get_kernel_client(kernel_id)
            try:
                yield kernel_client
            finally:
                await kernel_client.close()
        finally:
            await self.delete_kernel(kernel_id)


class JupyterGatewayContainer:
    DOCKERFILE = """\
FROM {base_image}

RUN pip install jupyter_kernel_gateway ipykernel

EXPOSE 8888
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888"]
"""

    def __init__(
        self,
        *,
        container_name: str | None = None,
        image_name: str = "jupyter-gateway",
        base_image: str = "jupyter/docker-stacks-foundation",
        startup_timeout: int = 60,
    ):
        self._client = aiodocker.Docker()
        self._image_name = image_name
        self._base_image = base_image
        self._startup_timeout = startup_timeout
        self._jupyter_port = random.randint(15000, 16000)
        self._jupyter_token = str(uuid.uuid4())
        self._container_name = container_name or f"{self._image_name}-{uuid.uuid4().hex[:8]}"

    @property
    def connection_info(self) -> JupyterConnectionInfo:
        return JupyterConnectionInfo(
            host="localhost", use_https=False, port=self._jupyter_port, token=self._jupyter_token
        )

    async def _build_container(self) -> None:
        logger.info(f"Building image {self._image_name} from {self._base_image} ...")

        dockerfile = JupyterGatewayContainer.DOCKERFILE.format(base_image=self._base_image)
        dockerfile_buffer = io.BytesIO(dockerfile.encode())
        tar_buffer = aiodocker.utils.mktar_from_dockerfile(dockerfile_buffer)

        await self._client.images.build(
            fileobj=tar_buffer,
            encoding="gzip",
            tag=self._image_name,
        )

        logger.success(f"Image built [{self._image_name}]")

    async def _start_container(self) -> None:
        logger.info(f"Starting container [{self._container_name}]")

        self._container = await self._client.containers.create(
            config={
                "Image": self._image_name,
                "Cmd": [
                    "jupyter",
                    "kernelgateway",
                    "--KernelGatewayApp.ip=0.0.0.0",
                    "--KernelGatewayApp.port=8888",
                    f'--KernelGatewayApp.auth_token="{self._jupyter_token}"',
                    "--JupyterApp.answer_yes=true",
                    "--JupyterWebsocketPersonality.list_kernels=true",
                ],
                "HostConfig": {
                    "PortBindings": {
                        "8888/tcp": [{"HostPort": str(self._jupyter_port)}],
                    }
                },
            },
            name=self._container_name,
        )

        await self._container.start()

        logger.info(f"Waiting for container ... [{self._container_name}]")

        start_time = time.time()

        while time.time() - start_time < self._startup_timeout:
            container_info = await self._container.show()
            if container_info["State"]["Status"] == "running":
                break
            await asyncio.sleep(1)
        else:
            raise TimeoutError(f"Container did not start within {self._startup_timeout} seconds")

        self.gateway_client = JupyterGatewayClient(self.connection_info)

        logger.info(f"Waiting for Kernel Gateway ... [{self._container_name}]")

        while time.time() - start_time < self._startup_timeout:
            if await self.gateway_client.is_ready():
                break
            await asyncio.sleep(1)
        else:
            raise TimeoutError(f"Kernel Gateway did not start within {self._startup_timeout} seconds")

        logger.success(f"Ready [{self._container_name}]")

    async def __aenter__(self) -> Self:
        await self._build_container()
        await self._start_container()

        return self

    async def __aexit__(self, exc_type: t.Any, exc_val: t.Any, exc_tb: t.Any) -> None:
        if self._container is not None:
            await self._container.stop(signal="SIGKILL")
            await self._container.delete()

        await self._client.close()

    @contextlib.asynccontextmanager
    async def create_kernel(self, kernel_name: str = "python3") -> t.AsyncGenerator[JupyterKernelClient, None]:
        if not self.gateway_client:
            raise RuntimeError("Kernel manager not initialized")

        available_kernels = await self.gateway_client.list_kernel_specs()
        if kernel_name not in available_kernels["kernelspecs"]:
            raise ValueError(f"Kernel {kernel_name} is not installed in {self._container_name}")

        async with self.gateway_client.create_kernel(kernel_name) as kernel_client:
            yield kernel_client


class Code(rg.Model):
    content: str


g_system_prompt = f"""
You are a helpful assistant with access to execute python code in a ipython kernel.

As you assist, emit any code you would like to execute in the following format:

{Code.xml_example()}

The results of code execution will be provided before continuing the conversation.
"""


async def main(generator_id: str) -> None:
    logger.success("Starting Jupyter example")
    logger.info(f"  |- Generator: {generator_id}")

    async with JupyterGatewayContainer() as container:
        async with container.create_kernel() as kernel:

            async def parse_and_run_code(chat: rg.Chat) -> rg.Chat | None:
                code = chat.last.try_parse(Code)
                if code is None:
                    return None

                logger.debug(f"Executing code:\n{code.content}")
                result = await kernel.execute(code.content)
                logger.debug(f"Result ({result.status}):\n{result.output}")

                return (
                    await chat.continue_(
                        f"""
                        <output status="{result.status}">
                        {result.output}
                        </output>
                        """
                    )
                    .then(parse_and_run_code)
                    .run()
                )

            pipeline = (
                rg.get_generator(generator_id)
                .chat({"role": "system", "content": g_system_prompt})
                .then(parse_and_run_code)
            )

            await rg.interact(pipeline)


@click.command()
@click.option(
    "-g",
    "--generator-id",
    type=str,
    default="claude-3-5-sonnet-20240620",
    required=True,
    help="Rigging generator identifier (gpt-4, mistral/mistral-medium, etc.)",
)
@click.option(
    "--log-level",
    type=click.Choice(logging.LogLevelList),
    default="info",
)
@click.option("--log-file", type=click.Path(path_type=pathlib.Path), default="jupyter.log")
@click.option(
    "--log-file-level",
    type=click.Choice(logging.LogLevelList),
    default="trace",
)
def cli(
    generator_id: str,
    log_level: logging.LogLevelLiteral,
    log_file: pathlib.Path,
    log_file_level: logging.LogLevelLiteral,
) -> None:
    """
    Rigging example of Jupyter interaction.
    """

    logging.configure_logging(log_level, log_file, log_file_level)
    asyncio.run(main(generator_id))


if __name__ == "__main__":
    cli()
