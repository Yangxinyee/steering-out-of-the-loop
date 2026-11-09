"""The sandbox around the agent's Python kernel (agent/kernel_main.py), (adapted from case 1's agent kit).

The agent's code is written by an external model, so it runs with nothing of this machine in reach:
  bubblewrap  every namespace unshared (no network, own PIDs, no other users' processes), a fresh /dev without GPUs,
              /tmp in memory; the only host paths inside are the system's /usr (read-only), the Python interpreter
              and the kernel's packages at /opt (read-only, so no host path appears in a traceback), the model files at
              /model and the training records at /data (read-only; synthetic), and the run's own work folder at /work
              (read-write)
  systemd     a user scope caps memory (default 32 GB, no swap) and CPU (default 8 cores)
  harness     one code run may take at most 900 s; past that, or when the kernel dies (memory), the kernel is killed
              and started again, and its variables are lost (the agent is told so)
  pressure    systemd-oomd kills processes of the whole user@<uid>.service once its memory pressure ("full" avg10)
              passes 50%. While code runs the pressure (the larger of "some" and "full" avg10) is read every
              second; this kernel is stopped by its PID above 45%, or after 4 minutes in a row above 30%, and handled
              like a kernel killed by the memory limit. Between turns the harness waits for the pressure to fall.
Every run records the kernel's memory peak in Kernel.last_stats: its scope's memory.peak (page cache included, as
oomd counts it) and the Python process's peak resident size (VmHWM), both reset at the start of each run.
"""
import json
import os
import select
import shutil
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VENV = REPO / ".venv-kernel"
PSI = f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/memory.pressure"
KILL_AT = 45.0                       # % (pressure level) while code runs; systemd-oomd acts at 50 (full avg10)
HIGH_AT, HIGH_S = 30.0, 240          # ... or above 30% for 4 minutes: the eICU queue kills its lanes after 5


def memory_pressure(path=PSI):
    """{"some": avg10, "full": avg10, "level": the larger of the two} of a cgroup's memory.pressure (percent), or None
    if it cannot be read; "level" is what the eICU queue's guard reads"""
    try:
        out = {}
        for line in open(path):
            kind, *fields = line.split()
            out[kind] = float(dict(f.split("=") for f in fields)["avg10"])
        out["level"] = max(out.values())
        return out
    except (OSError, ValueError, KeyError):
        return None


def _children(pid):
    try:
        return [int(c) for t in os.listdir(f"/proc/{pid}/task") for c in open(f"/proc/{pid}/task/{t}/children").read().split()]
    except OSError:
        return []


def interpreter_home(venv=VENV):
    """the uv-managed CPython the venv was made from (pyvenv.cfg "home" is its bin/)"""
    for line in (venv / "pyvenv.cfg").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "home":
            return Path(v.strip()).parent
    raise RuntimeError(f"no home in {venv}/pyvenv.cfg")


class Kernel:
    def __init__(self, work, model, data, mode, mem="32G", cpus=8, timeout=900, venv=VENV, psi=PSI, kill_at=KILL_AT):
        self.work, self.model, self.data, self.mode = Path(work), Path(model), Path(data), mode
        self.mem, self.cpus, self.timeout, self.venv = mem, cpus, timeout, Path(venv)
        self.psi, self.kill_at = psi, kill_at
        self.p, self.last_stats = None, {}

    def command(self):
        py = interpreter_home(self.venv)
        site = next(self.venv.glob("lib/python3*/site-packages"))
        ver = site.parent.name                                   # python3.12
        sandbox = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
                   "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                   "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
                   "--ro-bind", str(py), "/opt/python", "--ro-bind", str(site), "/opt/site",
                   "--ro-bind", str(HERE / "kernel_main.py"), "/kernel/kernel_main.py",
                   "--ro-bind", str(self.model), "/model", "--ro-bind", str(self.data), "/data",
                   "--bind", str(self.work), "/work", "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev",
                   "--ro-bind-try", "/sys/devices/system/cpu", "/sys/devices/system/cpu",
                   "--setenv", "PATH", "/opt/python/bin:/usr/bin", "--setenv", "HOME", "/tmp",
                   "--setenv", "PYTHONPATH", "/opt/site", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                   "--setenv", "LANG", "C.UTF-8", "--setenv", "OMP_NUM_THREADS", str(self.cpus), "--setenv", "AWR_MODE", self.mode,
                   "--chdir", "/work", f"/opt/python/bin/{ver}", "-u", "/kernel/kernel_main.py"]
        cap = []
        if self.mem and shutil.which("systemd-run"):
            cap = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={self.mem}", "-p", "MemorySwapMax=0",
                   "-p", f"CPUQuota={100 * self.cpus}%", "--"]
        return cap + sandbox

    def start(self):
        self.work.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        msg = self._read(300)
        if not msg or not msg.get("ready"):
            self.kill()
            raise RuntimeError(f"kernel did not start: {msg}")
        return msg

    def _scope(self):
        """the cgroup folder of the kernel's systemd scope, or None"""
        try:
            path = open(f"/proc/{self.p.pid}/cgroup").read().strip().split("::", 1)[1]
            return f"/sys/fs/cgroup{path}" if path.endswith(".scope") else None
        except (OSError, IndexError):
            return None

    def _peak_start(self):
        """memory.peak, reset for this file handle (Linux 6.12+): the scope's peak during the coming run"""
        self._peak_fd = None
        sc = self._scope()
        if sc:
            try:
                self._peak_fd = open(f"{sc}/memory.peak", "r+")
                self._peak_fd.write("reset\n")
                self._peak_fd.flush()
            except OSError:
                self._peak_fd = None

    def _peak_end(self):
        try:
            self._peak_fd.seek(0)
            v = int(self._peak_fd.read())
            self._peak_fd.close()
            return v
        except (OSError, ValueError, AttributeError):
            return None

    def _python_pid(self):
        """host PID of the kernel's Python process (bwrap's descendant)"""
        todo, seen = [self.p.pid], 0
        while todo and seen < 50:
            pid = todo.pop()
            seen += 1
            try:
                argv = open(f"/proc/{pid}/cmdline").read().split("\0")
                if "python" in os.path.basename(argv[0]) and any(a.endswith("kernel_main.py") for a in argv):
                    return pid
            except OSError:
                pass
            todo += _children(pid)
        return None

    def _hwm(self, reset=False):
        """the Python process's peak resident size (VmHWM) in bytes; reset=True starts a new peak (clear_refs 5)"""
        pid = self._python_pid()
        try:
            if reset:
                open(f"/proc/{pid}/clear_refs", "w").write("5")
                return None
            return int(open(f"/proc/{pid}/status").read().split("VmHWM:")[1].split()[0]) * 1024
        except (OSError, IndexError, ValueError, TypeError):
            return None

    def _read(self, timeout, watch=False):
        end = time.time() + timeout
        high_since = None
        while time.time() < end:
            r, _, _ = select.select([self.p.stdout], [], [], min(1, max(0.1, end - time.time())))
            if r:
                line = self.p.stdout.readline()
                return json.loads(line) if line else None          # "" = the kernel died
            if self.p.poll() is not None:
                return None
            if watch:
                pr = memory_pressure(self.psi)
                if pr is not None:
                    self.last_stats["pressure_max"] = max(self.last_stats.get("pressure_max", 0), pr["level"])
                    high_since = (high_since or time.time()) if pr["level"] > HIGH_AT else None
                    if pr["level"] > self.kill_at or (high_since and time.time() - high_since >= HIGH_S):
                        self.last_stats["pressure_kill"] = pr
                        return "pressure"
        return "timeout"

    def kill(self):
        if self.p and self.p.poll() is None:
            os.kill(self.p.pid, signal.SIGKILL)       # bwrap: its PID namespace, the kernel included, goes with it
            self.p.wait()
        self.p = None

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def run(self, code):
        """(output, restarted): restarted is a reason when the kernel had to be started again (its variables lost)"""
        restarted = None
        if not self.alive():
            self.start()
            restarted = "start"
        self.last_stats = {}
        self._hwm(reset=True)
        self._peak_start()
        self.p.stdin.write(json.dumps({"code": code}) + "\n")
        self.p.stdin.flush()
        msg = self._read(self.timeout, watch=True)
        peak = self._peak_end()
        if peak is not None:
            self.last_stats["mem_peak_gb"] = round(peak / 2 ** 30, 2)
        hwm = self._hwm()
        if hwm is not None:
            self.last_stats["python_peak_gb"] = round(hwm / 2 ** 30, 2)
        if msg == "timeout":
            self.kill()
            self.start()
            return f"[the run took longer than {self.timeout} seconds and was stopped]", "time limit"
        if msg == "pressure":
            self.kill()
            self.start()
            return ("[the kernel was stopped while running this code because the workstation ran short of memory]",
                    "memory pressure")
        if msg is None:
            self.kill()
            self.start()
            return f"[the kernel died while running this code (the memory limit is {self.mem}B)]", "memory limit or crash"
        return msg["output"], restarted
