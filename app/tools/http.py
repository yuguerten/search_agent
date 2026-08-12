from __future__ import annotations

import asyncio
from typing import Any

import httpx


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retries: int = 3,
    retry_statuses: set[int] | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Run a bounded retry loop for transient scholarly API failures."""

    retry_statuses = retry_statuses or {408, 429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = await client.request(method, url, **kwargs)
            if response.status_code not in retry_statuses:
                return response
            last_error = httpx.HTTPStatusError(
                f"Transient HTTP status {response.status_code}",
                request=response.request,
                response=response,
            )
            if attempt == retries - 1:
                return response
            retry_after = response.headers.get("retry-after")
            if retry_after:
                delay = float(retry_after)
            elif response.status_code == 429:
                delay = 10.0 * (attempt + 1)
            else:
                delay = 2**attempt
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            delay = 2**attempt
        await asyncio.sleep(min(delay, 60.0))

    if last_error:
        raise last_error
    raise RuntimeError("request retry loop ended unexpectedly")
