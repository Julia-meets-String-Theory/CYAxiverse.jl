"""HTTP range-request file object so pyarrow can read single parquet columns
from the HuggingFace KS mirror without downloading whole files (read-only)."""
import io
import requests

BASE = "https://huggingface.co/datasets/calabi-yau-data/polytopes-4d/resolve/main/"
REVISION_API = "https://huggingface.co/api/datasets/calabi-yau-data/polytopes-4d"


class RangeFile(io.RawIOBase):
    def __init__(self, url, session=None, block=1 << 20):
        self.url = url
        self.s = session or requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.final_url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self.cache = {}
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _get(self, start, end):
        for attempt in range(5):
            try:
                r = self.s.get(self.final_url, headers={"Range": f"bytes={start}-{end-1}"}, timeout=120)
                if r.status_code in (200, 206):
                    self.bytes_fetched += len(r.content)
                    return r.content
                if r.status_code in (401, 403):
                    # signed redirect expired: refresh
                    self.final_url = self.s.head(self.url, allow_redirects=True, timeout=60).url
            except requests.RequestException:
                pass
        raise IOError(f"range fetch failed {start}-{end}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block:
            data = self._get(self.pos, self.pos + n)
            self.pos += n
            return data
        out = bytearray()
        while n > 0:
            b = self.pos // self.block
            if b not in self.cache:
                start = b * self.block
                end = min(self.size, start + self.block)
                if len(self.cache) > 64:
                    self.cache.clear()
                self.cache[b] = self._get(start, end)
            chunk = self.cache[b]
            off = self.pos - b * self.block
            take = min(n, len(chunk) - off)
            out += chunk[off:off + take]
            self.pos += take
            n -= take
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_parquet(name, session=None):
    import pyarrow.parquet as pq
    f = RangeFile(BASE + name, session=session)
    return pq.ParquetFile(f), f
