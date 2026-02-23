import asyncio
import math
import sys

from schemas import Robot


class IPDiscovery:
    @staticmethod
    async def ping(ip: str, ping_timeout: float = 1.0) -> bool:
        """Async ping using the system ``ping`` command.

        Parameters
        ----------
        ip:
            The IPv4/IPv6 address or hostname to ping.
        ping_timeout:
            Maximum time **in seconds** to wait for a reply.

        Notes
        -----
        The ``-W`` flag semantics differ across platforms:

        * **Linux** — ``-W`` expects **seconds**.
        * **macOS** — ``-W`` expects **milliseconds**.
        * **Windows** — ``-w`` (lowercase) expects **milliseconds**.
        """
        is_windows = sys.platform.lower().startswith("win")
        count_flag = "-n" if is_windows else "-c"

        if is_windows:
            # Windows: -w <milliseconds>
            timeout_flag = "-w"
            timeout_value = str(int(ping_timeout * 1000))
        elif sys.platform == "darwin":
            # macOS: -W <milliseconds>
            timeout_flag = "-W"
            timeout_value = str(int(ping_timeout * 1000))
        else:
            # Linux: -W <seconds> (integer, rounded up)
            timeout_flag = "-W"
            timeout_value = str(math.ceil(ping_timeout))

        command = ["ping", count_flag, "1", timeout_flag, timeout_value, ip]

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return (await proc.wait()) == 0

    async def is_reachable(self, robot: Robot) -> bool:
        if not robot.connection_string:
            return False
        return await self.ping(robot.connection_string)
