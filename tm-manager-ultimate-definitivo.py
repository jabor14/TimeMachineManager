#!/opt/homebrew/bin/python3

import logging
import os
import plistlib
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk

# Setup basic logging
log_path = Path.home() / "tm_manager.log"
logging.basicConfig(
    level=logging.DEBUG,
    filename=str(log_path),
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tm_manager")

# Absolute paths to avoid PATH/TCC quirks when elevating via osascript
TMUTIL = "/usr/bin/tmutil"
DISKUTIL = "/usr/sbin/diskutil"


def format_bytes(num_bytes):
    """Return a short human-readable size string given bytes."""
    if num_bytes is None:
        return "N/A"
    try:
        num_bytes = float(num_bytes)
        if num_bytes >= 1024**4:
            return f"{num_bytes / (1024 ** 4):.1f}T"
        if num_bytes >= 1024**3:
            return f"{num_bytes / (1024 ** 3):.1f}G"
        if num_bytes >= 1024**2:
            return f"{num_bytes / (1024 ** 2):.1f}M"
        if num_bytes >= 1024:
            return f"{num_bytes / 1024:.1f}K"
        return f"{int(num_bytes)}B"
    except Exception:
        return "N/A"


def human_size_to_kb(size_str):
    """Convert human readable size like '1.2G', '345.6M', '123K' to integer KB.
    Returns None if cannot parse or 'N/A'."""
    try:
        if not size_str or size_str == "N/A":
            return None
        s = size_str.strip()
        if s.endswith("(estimado)"):
            s = s.replace("(estimado)", "").strip()
        if s.endswith("G"):
            val = float(s[:-1])
            return int(val * 1024 * 1024)
        if s.endswith("M"):
            val = float(s[:-1])
            return int(val * 1024)
        if s.endswith("K"):
            val = float(s[:-1])
            return int(val)
        # fallback: try to parse as bytes or raw number -> assume KB
        return int(float(s))
    except Exception:
        return None


class TMManager:
    def __init__(self):
        self.admin_authorized = False
        # cache snapshot sizes per volume to avoid repeated diskutil calls
        self._snapshot_size_cache = {}

    def run_tmutil(self, args, admin=False):
        import shlex

        cmd = [TMUTIL] + args
        logger.debug(f"Running tmutil: {' '.join(cmd)} admin={admin}")
        if admin:
            # build safe shell command and escape double quotes for AppleScript
            shell_command = " ".join(shlex.quote(c) for c in cmd)
            escaped = shell_command.replace('"', '\\"')
            osa_cmd = 'do shell script "' + escaped + '" with administrator privileges'
            result = subprocess.run(
                ["osascript", "-e", osa_cmd],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error(f"tmutil admin error: {result.stderr}")
                raise Exception(result.stderr)
            logger.debug(f"tmutil admin output: {result.stdout}")
            return result.stdout
        else:
            out = subprocess.check_output(cmd, text=True)
            logger.debug(f"tmutil output: {out}")
            return out

    def _run_as_admin_command(self, cmd_list):
        """Run arbitrary shell command (list) as admin via osascript and return stdout."""
        import shlex

        shell_command = " ".join(shlex.quote(c) for c in cmd_list)
        escaped = shell_command.replace('"', '\\"')
        osa_cmd = 'do shell script "' + escaped + '" with administrator privileges'
        result = subprocess.run(
            ["osascript", "-e", osa_cmd], capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Admin command error: {result.stderr}")
            raise Exception(result.stderr)
        return result.stdout

    def request_admin(self):
        """Trigger macOS admin prompt (osascript) to request privileges and cache authorization."""
        try:
            # benign command to ask for password
            out = self._run_as_admin_command(["echo", "tm_manager_admin_ok"])
            logger.info("Admin authorization granted")
            self.admin_authorized = True
            return True
        except Exception as e:
            logger.warning(f"Admin authorization failed: {e}")
            self.admin_authorized = False
            return False

    def list_snapshots_and_backups(self):
        # Snapshots locales: try multiple mount points so we detect snapshots that
        # live on other volumes (for instance /Volumes/Home). We will query
        # listlocalsnapshotdates and fall back to listlocalsnapshots for each candidate
        # mount and collect unique timestamps.
        # Reset snapshot size cache whenever we rebuild the list
        self._snapshot_size_cache = {}
        snap_dates = []
        seen = set()
        # Build candidate mount points: root, $HOME, and entries under /Volumes
        candidates = ["/"]
        # Also explicitly include /Volumes/Home which is a common mounted home volume
        # (helps when /Volumes listing is restricted or non-standard)
        try:
            if "/Volumes/Home" not in candidates:
                candidates.append("/Volumes/Home")
        except Exception:
            pass
        try:
            home = str(Path.home())
            if home and home not in candidates:
                candidates.append(home)
        except Exception:
            pass
        try:
            for p in sorted(os.listdir("/Volumes")):
                full = os.path.join("/Volumes", p)
                if full not in candidates:
                    candidates.append(full)
        except Exception:
            # if /Volumes is not accessible, ignore
            pass

        # First try to use diskutil via admin batch to get snapshot names and sizes
        if getattr(self, "admin_authorized", False):
            try:
                snap_info = self._collect_snapshot_info_admin(candidates)
                if snap_info:
                    # Pre-fill snapshot size cache and build snap_dates from keys
                    for (snap_date, vol), bytes_val in snap_info.items():
                        if snap_date not in seen:
                            snap_dates.append((snap_date, vol))
                            seen.add(snap_date)
                        # Populate snapshot size cache per volume
                        cache = self._snapshot_size_cache.get(vol)
                        if cache is None:
                            cache = {}
                            self._snapshot_size_cache[vol] = cache
                        cache[snap_date] = bytes_val
                else:
                    # if admin batch failed or returned nothing, fall back to tmutil/diskutil queries
                    raise Exception("snapshot admin batch returned empty")
            except Exception as e:
                logger.debug(
                    f"Snapshot admin batch failed or empty: {e}, falling back to tmutil"
                )
                snap_info = None

        # If snap_dates is still empty, fall back to tmutil listlocalsnapshotdates
        if not snap_dates:
            for cand in candidates:
                try:
                    snaps = self.run_tmutil(["listlocalsnapshotdates", cand])
                    for line in snaps.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("Snapshot dates for disk"):
                            continue
                        if line in seen:
                            continue
                        seen.add(line)
                        snap_dates.append((line, cand))
                    continue
                except Exception:
                    # try fallback listlocalsnapshots
                    try:
                        snaps2 = self.run_tmutil(["listlocalsnapshots", cand])
                        for line in snaps2.splitlines():
                            m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{6})", line)
                            if m:
                                val = m.group(1)
                                if val not in seen:
                                    seen.add(val)
                                    snap_dates.append((val, cand))
                    except Exception:
                        # ignore this candidate
                        pass
        # Backups externos
        backup_paths = []
        sizes_map = {}
        if getattr(self, "admin_authorized", False):
            try:
                sizes_map = self.list_backups_and_sizes_admin()
                backup_paths = list(sizes_map.keys())
            except Exception as e:
                logger.exception(f"Could not list backups and sizes via batch admin: {e}")
                sizes_map = {}
                backup_paths = []
        else:
            try:
                backups = self.run_tmutil(["listbackups"], admin=True)
                backup_paths = [
                    line.strip() for line in backups.strip().split("\n") if line.strip()
                ]
            except Exception as e:
                logger.error(f"Could not list external backups as admin: {e}")
                backup_paths = []

        def parse_fecha(fecha):
            try:
                return datetime.strptime(fecha, "%Y-%m-%d-%H%M%S")
            except Exception:
                return None

        # Prepara snapshots como datetimes
        dt_snaps = [
            (snap, parse_fecha(snap), vol)
            for snap, vol in snap_dates
            if parse_fecha(snap)
        ]

        # Prepara backups como datetimes
        backup_info = []
        for path in backup_paths:
            name = path.split("/")[-1]
            fecha_raw = name.replace(".backup", "")
            fecha_dt = parse_fecha(fecha_raw)
            if sizes_map and path in sizes_map:
                size = sizes_map.get(path, "N/A")
            else:
                size = self.get_backup_size(path)
            backup_info.append(
                {
                    "name": name,
                    "fecha": fecha_raw,
                    "fecha_dt": fecha_dt,
                    "type": "externo",
                    "ruta": path,
                    "size": size,
                    "estado": "Backup externo",
                }
            )

        # Vinculación tolerante ±10 minutos
        def vinculacion_fecha(dt1, dt2):
            if not dt1 or not dt2:
                return False
            return abs((dt1 - dt2).total_seconds()) <= 10 * 60

        items = []
        for snap, snap_dt, vol in dt_snaps:
            linked_backup = ""
            for b in backup_info:
                if vinculacion_fecha(snap_dt, b["fecha_dt"]):
                    linked_backup = b["ruta"] or b.get("name", "")
                    break
            vinculado = bool(linked_backup)
            items.append(
                {
                    "nombre": snap,
                    "fecha": snap,
                    "type": "local",
                    "ruta": f"Snapshot APFS: {snap}",
                    "size": self.get_snapshot_size(snap, vol),
                    "estado": "Snapshot local",
                    "vinculado": vinculado,
                    "linked_backup": linked_backup,
                }
            )
        for b in backup_info:
            linked_snap = ""
            for snap, snap_dt, vol in dt_snaps:
                if vinculacion_fecha(b["fecha_dt"], snap_dt):
                    linked_snap = snap
                    break
            vinculado = bool(linked_snap)
            # Use the size computed (could be from batch sizes_map or per-path get_backup_size)
            items.append(
                {
                    "nombre": b["name"],
                    "fecha": b["fecha"],
                    "type": "externo",
                    "ruta": b["ruta"],
                    "size": b.get("size", "N/A"),
                    "estado": b["estado"],
                    "vinculado": vinculado,
                    "linked_backup": linked_snap,
                }
            )

        items.sort(key=lambda x: x["fecha"], reverse=True)
        return items

    def get_snapshot_size(self, snap_date, volume="/"):
        """Return the actual APFS snapshot size when available, falling back to an estimate."""
        try:
            cache = self._snapshot_size_cache.get(volume)
            if cache is None:
                cache = self._collect_snapshot_sizes(volume)
                self._snapshot_size_cache[volume] = cache
            if cache:
                size_bytes = cache.get(snap_date)
                if size_bytes is not None:
                    return format_bytes(size_bytes)
        except Exception as e:
            logger.exception(
                f"Failed to get snapshot size for {snap_date} on {volume}: {e}"
            )

        # If we reach here, fall back to the legacy estimation logic
        return self._estimate_snapshot_size(snap_date, volume)

    def _collect_snapshot_sizes(self, volume):
        """Return a dict {snapshot_date: bytes_used} for the given volume.
        Works across macOS locale changes (Tahoe 26) and handles new key names.
        """
        sizes = {}
        cmd = [DISKUTIL, "apfs", "listSnapshots", "-plist", volume]

        def _run_diskutil_bytes():
            # Return raw bytes output (not text) so plistlib can parse either XML or binary plists
            try:
                completed = subprocess.run(
                    cmd, capture_output=True, text=False, timeout=120
                )
                if completed.returncode != 0:
                    raise Exception(completed.stderr.decode("utf-8", "ignore"))
                return completed.stdout
            except Exception as e:
                if getattr(self, "admin_authorized", False):
                    try:
                        # Re-run elevated
                        out_txt = self._run_as_admin_command(cmd)
                        return out_txt.encode("utf-8", "ignore")
                    except Exception as admin_e:
                        raise Exception(str(admin_e)) from admin_e
                raise

        try:
            raw = _run_diskutil_bytes()
            if not raw:
                return sizes
            try:
                plist = plistlib.loads(raw)
            except Exception:
                # If for some reason plist parsing fails, fall back to text mode
                txt = raw.decode("utf-8", "ignore")
                return self._parse_listSnapshots_text(txt)

            snapshots = plist.get("Snapshots") or []
            for snap in snapshots:
                name = snap.get("SnapshotName", "") or snap.get("Name", "")
                # Accept several possible keys across macOS versions
                bytes_used = (
                    snap.get("BytesUsed")
                    or snap.get("SizeInBytes")
                    or snap.get("SnapshotBytesUsed")
                    or snap.get("SnapshotDiskUsage")
                )
                if isinstance(bytes_used, str):
                    # Remove thousand separators and any non-digits
                    digits = re.sub(r"[^0-9]", "", bytes_used)
                    bytes_used = int(digits) if digits else None
                date_match = None
                if name:
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{6})", name)
                if date_match and bytes_used is not None:
                    try:
                        sizes[date_match.group(1)] = int(bytes_used)
                    except Exception:
                        try:
                            sizes[date_match.group(1)] = int(float(bytes_used))
                        except Exception:
                            logger.debug(
                                f"Could not parse bytes for snapshot {name}: {bytes_used}"
                            )
            # If nothing parsed, fall back to plain-text parsing (localized output)
            if not sizes:
                txt = raw.decode("utf-8", "ignore")
                sizes = self._parse_listSnapshots_text(txt)
            return sizes
        except Exception as e:
            logger.warning(f"Failed to collect snapshot sizes for {volume}: {e}")
            return sizes

    def _parse_listSnapshots_text(self, txt):
        """Parse non-plist `diskutil apfs listSnapshots` localized output.
        Returns {snapshot_date: bytes}.
        """
        sizes: dict[str, int] = {}
        current_name = None
        for line in (txt or "").splitlines():
            # Normalize
            l = line.strip()
            if not l:
                continue
            # Name/Nombre/Nom
            m_name = re.search(r"(?:Name|Nombre|Nom)\s*:\s*(\S+)", l, re.IGNORECASE)
            if m_name:
                current_name = m_name.group(1)
                continue
            # Bytes Used / Bytes usados / Bytes utilitzats
            m_bytes = re.search(
                r"(?:Bytes\s*Used|Bytes\s*usados?|Bytes\s*utilitzats?)\s*:\s*([0-9\., ]+)",
                l,
                re.IGNORECASE,
            )
            if m_bytes and current_name:
                digits = re.sub(r"[^0-9]", "", m_bytes.group(1))
                if digits:
                    try:
                        b = int(digits)
                    except Exception:
                        try:
                            b = int(float(digits))
                        except Exception:
                            b = None
                    if b is not None:
                        m_date = re.search(r"(\d{4}-\d{2}-\d{2}-\d{6})", current_name)
                        if m_date:
                            sizes[m_date.group(1)] = b
        return sizes

    def _estimate_snapshot_size(self, snap_date, volume="/"):
        """Estimate snapshot size when diskutil cannot provide exact bytes.
        Uses `diskutil apfs query-reclaimable-space` and is robust to localization.
        """
        try:
            cmd = [DISKUTIL, "apfs", "query-reclaimable-space", volume]
            try:
                completed = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                if completed.returncode != 0:
                    raise Exception(completed.stderr)
                out = completed.stdout
            except Exception as e:
                logger.warning(
                    f"diskutil query-reclaimable-space failed for {volume}: {e}, trying with admin"
                )
                if getattr(self, "admin_authorized", False):
                    try:
                        out = self._run_as_admin_command(cmd)
                    except Exception as admin_e:
                        logger.error(
                            f"diskutil as admin failed for {volume}: {admin_e}"
                        )
                        return "N/A (estimado)"
                else:
                    return "N/A (estimado)"

            # Match both English/Spanish/Catalan like: "Reclaimable space: 12345 bytes" / "Espacio recuperable: 12345 bytes" / "Espai recuperable: 12345 bytes"
            m = re.search(
                r"(?:Reclaimable\s*space|Espacio\s*recuperable|Espai\s*recuperable)\s*:\s*([0-9\., ]+)\s*bytes",
                out,
                re.IGNORECASE,
            )
            if not m:
                # Fallback: first big integer in output
                m = re.search(r"([0-9][0-9\., ]+)\s*bytes", out, re.IGNORECASE)
            if not m:
                logger.warning("Could not parse reclaimable space from diskutil output")
                return "N/A (estimado)"
            digits = re.sub(r"[^0-9]", "", m.group(1))
            total_reclaimable_bytes = int(digits) if digits else 0

            try:
                snaps_out = self.run_tmutil(["listlocalsnapshotdates", volume])
            except Exception:
                snaps_out = ""
            num_snapshots = len(
                [
                    line
                    for line in snaps_out.splitlines()
                    if line.strip() and not line.startswith("Snapshot dates")
                ]
            )
            if num_snapshots == 0:
                return "0K (estimado)"
            avg_size_bytes = total_reclaimable_bytes / num_snapshots
            return f"{format_bytes(avg_size_bytes)} (estimado)"
        except Exception as e:
            logger.exception(f"Failed to estimate snapshot size for {snap_date}: {e}")
            return "N/A (estimado)"

    def get_backup_size(self, path):
        """Get backup unique size using tmutil uniquesize.
        Falls back to `du -sk` if tmutil output cannot be parsed.
        Returns a human-readable string.
        """
        try:
            logger.debug(f"Getting unique size for: {path}")

            candidates = [path]
            try:
                parent = os.path.dirname(path)
                if parent and parent not in candidates:
                    candidates.append(parent)
                grand = os.path.dirname(parent)
                if grand and grand not in candidates:
                    candidates.append(grand)
            except Exception:
                pass

            out = ""
            for p_try in candidates:
                cmd = [TMUTIL, "uniquesize", p_try]
                try:
                    completed = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=300
                    )
                    if completed.returncode == 0 and completed.stdout.strip():
                        out = completed.stdout
                    elif getattr(self, "admin_authorized", False):
                        out = self._run_as_admin_command(cmd)
                    if out.strip():
                        break
                except Exception as e:
                    logger.debug(f"tmutil uniquesize attempt failed for {p_try}: {e}")

            if out.strip():
                # tmutil may return one or more lines. Use the first number we see on the line
                first_line = out.strip().splitlines()[0]
                m = re.search(r"([0-9][0-9\.,]*)", first_line)
                if m:
                    num = m.group(1)
                    num = re.sub(r"[^0-9\.]", "", num)
                    try:
                        bytes_val = float(num)
                    except Exception:
                        bytes_val = float(int(num)) if num else 0.0

                    if bytes_val >= 1024 * 1024 * 1024:
                        return f"{bytes_val / (1024*1024*1024):.1f}G"
                    elif bytes_val >= 1024 * 1024:
                        return f"{bytes_val / (1024*1024):.1f}M"
                    elif bytes_val >= 1024:
                        return f"{bytes_val / 1024:.1f}K"
                    else:
                        return f"{int(bytes_val)}B"

            # Fallback: du -sk (size on disk, not unique) just so UI shows something useful
            du_out = ""
            try:
                du_cmd = ["/usr/bin/du", "-sk", path]
                if getattr(self, "admin_authorized", False):
                    du_out = self._run_as_admin_command(du_cmd)
                else:
                    completed = subprocess.run(
                        du_cmd, capture_output=True, text=True, timeout=300
                    )
                    du_out = completed.stdout if completed.returncode == 0 else ""
                if du_out.strip():
                    kb = int(re.split(r"\s+", du_out.strip())[0])
                    return (
                        f"{kb/1024/1024:.1f}G"
                        if kb >= 1024 * 1024
                        else (f"{kb/1024:.1f}M" if kb >= 1024 else f"{kb}K")
                    )
            except Exception as e:
                logger.debug(f"du fallback failed for {path}: {e}")

            logger.error(
                f"Failed to get backup size for {path}: tmutil/du returned no usable output."
            )
            return "N/A"
        except Exception as e:
            logger.exception(f"Failed to get backup size for {path}: {e}")
            return "N/A"

    def get_backup_sizes_batch_admin(self, paths):
        """Batch-fetch unique sizes for multiple backup paths with a SINGLE admin prompt.
        Returns a dict {path: human_readable_size}.
        Implementation: one elevated /bin/bash -lc script that loops paths and prints
        tab-separated lines: <path>\t<bytes> (falls back to du when tmutil fails).
        """
        try:
            if not paths:
                return {}
            if not getattr(self, "admin_authorized", False):
                # Trigger admin prompt once; if refused, we fall back later
                ok = self.request_admin()
                if not ok:
                    logger.warning(
                        "Admin authorization not granted; skipping batch sizes"
                    )
                    return {}

            # Build a safe bash array with quoted paths
            array_elems = []
            for p in paths:
                if not p:
                    continue
                # Avoid newlines; escape single quotes for bash $'...'
                q = p.replace("'", "'\\''")
                array_elems.append(f"'{q}'")
            if not array_elems:
                return {}
            bash_array = " ".join(array_elems)

            script = (
                "set -euo pipefail; "
                "paths=(" + bash_array + "); "
                'for p in "${paths[@]}"; do '
                '  [[ -z "$p" ]] && continue; '
                '  if [[ -e "$p" ]]; then '
                '    out=""; bytes=""; '
                f'    if out=$({TMUTIL} uniquesize "$p" 2>/dev/null | head -n1); then '
                "      bytes=$(echo \"$out\" | LC_ALL=C sed -E 's/[^0-9\\.]*//g'); "
                '      if [[ -z "$bytes" ]]; then '
                "        kb=$( /usr/bin/du -sk \"$p\" 2>/dev/null | awk '{print $1}' ); bytes=$((kb*1024)); "
                "      fi; "
                '      echo -e "$p\t$bytes"; '
                "    else "
                '      kb=$( /usr/bin/du -sk "$p" 2>/dev/null | awk \'{print $1}\' ); echo -e "$p\t$((kb*1024))"; '
                "    fi; "
                "  fi; "
                "done"
            )

            out = self._run_as_admin_command(["/bin/bash", "-lc", script])
            sizes = {}
            for line in (out or "").splitlines():
                try:
                    path, num = line.split("\t", 1)
                    num = num.strip()
                    # Allow decimal; interpret as bytes
                    num_digits = re.sub(r"[^0-9\.]", "", num)
                    bytes_val = float(num_digits) if num_digits else 0.0
                    sizes[path] = format_bytes(bytes_val)
                except Exception:
                    continue
            return sizes
        except Exception as e:
            logger.exception(f"Batch admin sizes failed: {e}")
            return {}

    def get_multiple_backup_sizes_admin(self, paths):
        """Return {path: size_str} using a single elevated batch call when possible.
        Falls back to per-path get_backup_size() if batching fails or is empty.
        """
        if not paths:
            return {}
        # First try the batch mode (one admin prompt)
        sizes = self.get_backup_sizes_batch_admin(paths)
        if sizes:
            # Ensure every requested path has a value
            for p in paths:
                if p not in sizes:
                    try:
                        sizes[p] = self.get_backup_size(p)
                    except Exception:
                        sizes[p] = "N/A"
            return sizes
        # Fallback: per-path (may prompt more often)
        sizes = {}
        for p in paths:
            try:
                sizes[p] = self.get_backup_size(p)
            except Exception:
                sizes[p] = "N/A"
        return sizes

    def _collect_snapshot_info_admin(self, volumes):
        """Collect snapshot names and sizes for multiple APFS volumes with a single elevated command.

        This helper runs a bash script as admin that loops over the provided volumes,
        invokes `diskutil apfs listSnapshots -plist` on each, and prints a volume marker
        before each plist. The output is then parsed into a mapping of
        `(snapshot_date, volume) -> bytes_used`. If a plist cannot be parsed, the entry
        for that volume is skipped. Requires self.admin_authorized to be True.

        Parameters
        ----------
        volumes : list[str]
            List of volume mount points (e.g. '/', '/Volumes/Home').

        Returns
        -------
        dict
            Mapping of (snapshot_date, volume) to bytes used (int).
        """
        try:
            if not volumes:
                return {}
            # Build volume array for bash script, quoting each path
            vol_elems = []
            for v in volumes:
                if not v:
                    continue
                q = v.replace("'", "'\\''")
                vol_elems.append(f"'{q}'")
            if not vol_elems:
                return {}
            vols_str = " ".join(vol_elems)
            # Construct a single bash script that prints a marker before each volume's snapshot plist
            script_lines = [
                "set -euo pipefail",
                f"vols=({vols_str})",
                'for v in "${vols[@]}"; do',
                '  echo "__VOL__${v}"',
                f'  {DISKUTIL} apfs listSnapshots -plist "${{v}}" 2>/dev/null || true',
                "done",
            ]
            script = "\n".join(script_lines) + "\n"
            out = self._run_as_admin_command(["/bin/bash", "-lc", script])
            if not out:
                return {}
            result = {}
            # Split output by volume markers
            segments = out.split("__VOL__")
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                # First line is the volume path
                first_nl = seg.find("\n")
                if first_nl == -1:
                    continue
                volume = seg[:first_nl].strip()
                plist_text = seg[first_nl + 1 :]
                if not plist_text.strip():
                    continue
                # Try to parse plist (could be XML or binary)
                try:
                    plist = plistlib.loads(plist_text.encode("utf-8", "ignore"))
                except Exception:
                    # fallback: treat as text and skip
                    continue
                snaps = plist.get("Snapshots") or []
                for snap in snaps:
                    name = snap.get("SnapshotName") or snap.get("Name") or ""
                    if not name:
                        continue
                    # Extract snapshot date (YYYY-MM-DD-HHMMSS) from the snapshot name
                    m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{6})", name)
                    if not m:
                        continue
                    snap_date = m.group(1)
                    bytes_used = (
                        snap.get("BytesUsed")
                        or snap.get("SizeInBytes")
                        or snap.get("SnapshotBytesUsed")
                        or snap.get("SnapshotDiskUsage")
                    )
                    if isinstance(bytes_used, str):
                        digits = re.sub(r"[^0-9]", "", bytes_used)
                        bytes_used = int(digits) if digits else None
                    try:
                        bytes_int = int(bytes_used)
                    except Exception:
                        try:
                            bytes_int = int(float(bytes_used))
                        except Exception:
                            bytes_int = None
                    if bytes_int is not None:
                        result[(snap_date, volume)] = bytes_int
            return result
        except Exception as e:
            logger.exception(f"Failed to collect snapshot info via admin script: {e}")
            return {}

    def list_backups_and_sizes_admin(self):
        """Return a dict mapping backup path to human-readable unique size in a single elevated command.

        This runs both `tmutil listbackups` and `tmutil uniquesize` in one script to minimise the number of
        admin prompts (one per call). It falls back to `du -sk` when uniquesize fails to return bytes.
        Requires self.admin_authorized to be True or will attempt to request admin authorization.
        """
        try:
            # Ensure admin authorization is available; request if needed
            if not getattr(self, "admin_authorized", False):
                ok = self.request_admin()
                if not ok:
                    logger.warning("Admin authorization not granted; cannot list backups with sizes")
                    return {}
            # Build a bash script that lists backups and computes unique size or disk usage
            script = (
                "set -euo pipefail\n"
                f"list=$({TMUTIL} listbackups 2>/dev/null || true)\n"
                "IFS=

    def resolve_backup_path(self, path):
        """Resolve a backup identifier (either full path or basename) to a full backup path.

        Strategy:
        - If path contains '/', assume it's already a full path and return it.
        - Otherwise, call `tmutil listbackups` and try matches in order:
          1) basename equality
          2) line endswith(path)
          3) line contains path
        - For any candidate, prefer the one where os.path.exists(candidate) is True.
        - Return the first good candidate, or None if none found.
        """
        import os

        try:
            if "/" in (path or ""):
                return path
            try:
                # Use admin to ensure we can see all backups to resolve the path
                backups_out = self.run_tmutil(["listbackups"], admin=True)
            except Exception as e:
                logger.debug(f"Could not list backups to resolve {path}: {e}")
                return None

            candidates = []
            for line in backups_out.splitlines():
                line = line.strip()
                if not line:
                    continue
                # basename equality
                try:
                    if os.path.basename(line) == path:
                        candidates.insert(0, line)
                        continue
                except Exception:
                    pass
                # endswith
                if line.endswith(path):
                    candidates.append(line)
                    continue
                # contains
                if path in line:
                    candidates.append(line)

            # Prefer existing paths
            for c in candidates:
                try:
                    if os.path.exists(c):
                        return c
                except Exception:
                    pass

            # fallback to first candidate if any
            if candidates:
                return candidates[0]
        except Exception:
            logger.exception(f"Error resolving backup path for {path}")
        return None

    def _find_previous_for_backup(self, marker_path):
        """Given a path like /Volumes/.timemachine/<UUID>/<name>.backup, try to
        find the actual '.previous' directory on mounted backup volumes.

        Strategy:
        - Extract the basename (e.g. 2025-10-01-141748.backup -> 2025-10-01-141748)
        - Look under each mount point in /Volumes (excluding .timemachine) for a
          directory named <basename>.previous and return the first existing path.
        - Return None if not found.
        """
        import os

        try:
            base = os.path.basename(marker_path)
            if base.endswith(".backup"):
                name = base.replace(".backup", "")
            else:
                name = base
            candidate_name = f"{name}.previous"
            for entry in sorted(os.listdir("/Volumes")):
                if entry == ".timemachine":
                    continue
                mountp = os.path.join("/Volumes", entry)
                try:
                    cand = os.path.join(mountp, candidate_name)
                    if os.path.exists(cand):
                        return cand
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def delete_snapshot(self, date):
        logger.info(f"Deleting local snapshot: {date}")
        return self.run_tmutil(["deletelocalsnapshots", date], admin=True)

    def delete_backup(self, path):
        logger.info(f"Deleting backup: {path}")
        try:
            resolved = self.resolve_backup_path(path)
            if resolved:
                logger.debug(f"Resolved backup {path} -> {resolved}")
                path = resolved
            else:
                logger.debug(
                    f"Could not resolve backup path for {path}; proceeding with original value"
                )
        except Exception:
            logger.exception(f"Error while resolving backup path for deletion: {path}")

        return self.run_tmutil(["delete", path], admin=True)


class TMManagerGUI(Gtk.Window):
    def __init__(self):
        super().__init__(title="Time Machine Manager Ultimate DEFINITIVO")
        self.set_default_size(1000, 700)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.manager = TMManager()
        # Run a startup TCC check to detect if the running Python lacks Full Disk Access
        self.create_main_ui()
        # Ask for admin privileges after the main UI is shown so the dialog is visible
        # and not blocking the constructor. This schedules a one-shot call on the
        # main loop to show the prompt once the window is visible.
        GLib.idle_add(self._ask_for_admin_dialog)
        # perform the check after UI is ready so we can show dialogs
        self._startup_tcc_check()

    def _ask_for_admin_dialog(self):
        """Show the initial admin request dialog once the UI is ready.

        Returned False to run only once when scheduled with GLib.idle_add.
        """
        try:
            dialog = Gtk.MessageDialog(
                parent=self,
                flags=0,
                message_type=Gtk.MessageType.QUESTION,
                buttons=Gtk.ButtonsType.YES_NO,
                text="Pedir permisos de administrador",
            )
            dialog.format_secondary_text(
                "¿Deseas conceder permisos de administrador para que la app pueda calcular tamaños y borrar backups sin restricciones? (Se pedirá la contraseña)"
            )
            response = dialog.run()
            dialog.destroy()
            if response == Gtk.ResponseType.YES:
                ok = self.manager.request_admin()
                self.manager.admin_authorized = bool(ok)
        except Exception:
            # If anything goes wrong showing the dialog, log and continue silently
            logger.exception("Failed to show initial admin dialog")
        return False

    def _startup_tcc_check(self):
        import shlex
        import sys

        def check_thread():
            # pick a sample path: prefer first external backup if available
            sample = None
            try:
                it = self.liststore.get_iter_first()
                while it:
                    item = self.liststore.get_value(it, 9)
                    if item and item.get("type") == "externo":
                        sample = item.get("ruta")
                        break
                    it = self.liststore.iter_next(it)
            except Exception:
                sample = None
            if not sample:
                sample = "/"

            logger.debug(
                f"Startup TCC check: testing ls on {sample} using {sys.executable}"
            )
            try:
                completed = subprocess.run(
                    ["ls", "-l", sample], capture_output=True, text=True, timeout=20
                )
                logger.debug(
                    f"Startup ls rc={completed.returncode} stdout={completed.stdout!r} stderr={completed.stderr!r}"
                )
                # Consider Operation not permitted or non-zero return code with stderr as failure
                if completed.returncode != 0 or (
                    "Operation not permitted" in (completed.stderr or "")
                    or "Permission denied" in (completed.stderr or "")
                ):
                    # Show dialog on main thread
                    GLib.idle_add(
                        self._show_tcc_dialog,
                        sys.executable,
                        completed.returncode,
                        completed.stderr,
                    )
            except Exception as e:
                logger.exception(f"Startup ls test failed: {e}")
                GLib.idle_add(self._show_tcc_dialog, sys.executable, -1, str(e))

        t = threading.Thread(target=check_thread)
        t.daemon = True
        t.start()

    def _show_tcc_dialog(self, py_executable, rc, stderr_text):
        # Inform the user that Full Disk Access may be needed and show actionable steps
        msg = f"El intérprete Python en uso es:\n{py_executable}\n\nResultado del test de permisos: rc={rc}\n\nErrores:\n{(stderr_text or '')[:1000]}\n\nSi ves 'Operation not permitted' o 'Permission denied', añade el ejecutable anterior a Preferencias → Privacidad y seguridad → Acceso completo al disco y reinicia la aplicación. ¿Abrir Preferencias ahora?"
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text="Full Disk Access probablemente requerido",
        )
        dialog.format_secondary_text(msg)
        resp = dialog.run()
        dialog.destroy()
        if resp == Gtk.ResponseType.OK:
            try:
                subprocess.run(
                    [
                        "open",
                        "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
                    ],
                    check=False,
                )
            except Exception:
                try:
                    subprocess.run(
                        ["open", "/System/Applications/System Settings.app"],
                        check=False,
                    )
                except Exception:
                    pass
        return False

    def create_main_ui(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        refresh_btn = Gtk.Button(label="Refrescar")
        refresh_btn.connect("clicked", self.on_refresh)
        button_box.pack_start(refresh_btn, False, False, 0)
        info_btn = Gtk.Button(label="Información")
        info_btn.connect("clicked", self.on_info)
        button_box.pack_start(info_btn, False, False, 0)
        diag_btn = Gtk.Button(label="Diagnóstico")
        diag_btn.connect("clicked", self.on_diag)
        button_box.pack_start(diag_btn, False, False, 0)
        help_btn = Gtk.Button(label="Ayuda")
        help_btn.connect("clicked", self.on_help)
        button_box.pack_start(help_btn, False, False, 0)
        # Request admin button (user can ask for admin later)
        self.request_admin_btn = Gtk.Button(label="Pedir permisos admin")
        self.request_admin_btn.connect("clicked", self.on_request_admin)
        button_box.pack_start(self.request_admin_btn, False, False, 0)
        # Full Disk Access help button
        fda_btn = Gtk.Button(label="Permisos de disco (Full Disk Access)")
        fda_btn.connect("clicked", self.on_open_full_disk_help)
        button_box.pack_start(fda_btn, False, False, 0)
        # Recalculate sizes button (uses admin batch if available)
        self.recalc_btn = Gtk.Button(label="Recalcular tamaños")
        self.recalc_btn.connect("clicked", self.on_recalc_sizes)
        button_box.pack_start(self.recalc_btn, False, False, 0)
        self.delete_btn = Gtk.Button(label="Borrar seleccionado")
        self.delete_btn.connect("clicked", self.on_delete)
        button_box.pack_end(self.delete_btn, False, False, 0)
        # Delete all button
        self.delete_all_btn = Gtk.Button(label="Borrar todo")
        self.delete_all_btn.connect("clicked", self.on_delete_all)
        button_box.pack_end(self.delete_all_btn, False, False, 0)
        # Exit button
        exit_btn = Gtk.Button(label="Salir")
        exit_btn.connect("clicked", lambda b: Gtk.main_quit())
        button_box.pack_end(exit_btn, False, False, 0)
        vbox.pack_start(button_box, False, False, 0)
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(500)
        # Use a TreeView with a ListStore to display a formatted table
        # Columns: Icon, Tipo, Nombre (markup), Fecha, Tamaño, Estado, Vinculado, Ruta (hidden), bg_color, item(dict)
        # Additional hidden columns for sorting and linked backup:
        # fecha_ts (float) used to sort by date, size_kb (int) used to sort by size, linked_backup (str)
        # bg_color is used to set per-row background (e.g. linked or backup rows)
        self.liststore = Gtk.ListStore(
            str, str, str, str, str, str, str, str, str, object, float, int, str
        )
        self.treeview = Gtk.TreeView(model=self.liststore)
        self.treeview.set_activate_on_single_click(True)

        def add_column(title, col_id, use_markup=False, visible=True, set_bg=False):
            renderer = Gtk.CellRendererText()
            if use_markup:
                column = Gtk.TreeViewColumn(title, renderer)
                column.add_attribute(renderer, "markup", col_id)
            else:
                column = Gtk.TreeViewColumn(title, renderer, text=col_id)
            if set_bg:
                # bind the cell background to the bg_color column (index 8)
                column.add_attribute(renderer, "cell-background", 8)
            column.set_visible(visible)
            column.set_resizable(True)
            self.treeview.append_column(column)
            return column

        # Icon renderer (simple letter)
        icon_renderer = Gtk.CellRendererText()
        icon_renderer.props.xalign = 0.5
        icon_column = Gtk.TreeViewColumn("", icon_renderer, text=0)
        icon_column.set_resizable(False)
        icon_column.set_min_width(48)
        self.treeview.append_column(icon_column)

        add_column("Tipo", 1)
        # Name column: use larger markup for better readability
        add_column("Nombre", 2, use_markup=True, set_bg=True)
        fecha_col = add_column("Fecha", 3, set_bg=True)
        size_col = add_column("Tamaño", 4, set_bg=True)
        add_column("Estado", 5, set_bg=True)
        add_column("Vinculado", 6, set_bg=True)
        # Ruta (oculta por defecto, pero disponible en el modelo)
        add_column("Ruta", 7, visible=False)
        # Linked backup (show basename or short path)
        linked_col = add_column("Backup vinculado", 12, visible=True)

        # Enable sorting: Fecha by fecha_ts (index 10), Tamaño by size_kb (index 11)
        try:
            fecha_col.set_sort_column_id(10)
            size_col.set_sort_column_id(11)
            # also mark linked_col sortable by its text value (index 12)
            linked_col.set_sort_column_id(12)
        except Exception:
            # if something goes wrong, continue without raising
            logger.exception("Could not set sort columns")

        # Selection handler to show ruta in status bar
        selection = self.treeview.get_selection()
        selection.connect("changed", self.on_selection_changed)

        # Enable query tooltip for showing ruta on hover
        self.treeview.set_has_tooltip(True)
        self.treeview.connect("query-tooltip", self.on_query_tooltip)

        scrolled.add(self.treeview)
        vbox.pack_start(scrolled, True, True, 0)
        self.status_label = Gtk.Label()
        self.status_label.set_halign(Gtk.Align.START)
        vbox.pack_start(self.status_label, False, False, 0)
        # Progress area for recalculating sizes
        progress_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_show_text(True)
        self.progress_bar.set_visible(False)
        progress_box.pack_start(self.progress_bar, True, True, 0)
        self.progress_cancel_btn = Gtk.Button(label="Cancelar")
        self.progress_cancel_btn.connect("clicked", self.on_cancel_recalc)
        self.progress_cancel_btn.set_visible(False)
        progress_box.pack_start(self.progress_cancel_btn, False, False, 0)
        vbox.pack_start(progress_box, False, False, 0)
        self._recalc_cancelled = False
        # Context menu for rows (right click)
        self.menu = Gtk.Menu()
        mi_info = Gtk.MenuItem(label="Información")
        mi_info.connect("activate", self.menu_info_activate)
        self.menu.append(mi_info)
        mi_delete = Gtk.MenuItem(label="Borrar")
        mi_delete.connect("activate", self.menu_delete_activate)
        self.menu.append(mi_delete)
        self.menu.show_all()

        # Double-click (row-activated) to show info
        self.treeview.connect("row-activated", self.on_row_activated)
        # Right click to show context menu
        self.treeview.connect("button-press-event", self.on_treeview_button_press)
        self.add(vbox)
        self.load_unified()

    def load_unified(self):
        # Clear liststore
        try:
            self.liststore.clear()
        except Exception:
            pass
        self.status_label.set_text("Cargando lista...")

        def load_thread():
            items = self.manager.list_snapshots_and_backups()
            GLib.idle_add(self.update_unified_list, items)

        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()

    def update_unified_list(self, items):
        if not items:
            self.status_label.set_text("No se encontraron snapshots ni backups")
            return

        # Populate liststore with formatted values. Use markup for Nombre to highlight it.
        for item in items:
            icon = "L" if item["type"] == "local" else "E"
            tipo = "Local" if item["type"] == "local" else "Externo"
            # Highlight linked items with a colored name
            if item.get("vinculado"):
                nombre_markup = (
                    f"<span foreground=\"#008000\"><b>{item['nombre']}</b></span>"
                )
                bg_color = "#f0fff0"  # light green
            else:
                nombre_markup = f"<b>{item['nombre']}</b>"
                bg_color = "#ffffff"
            fecha = item.get("fecha", "")
            size = item.get("size", "")
            estado = item.get("estado", "")
            vinculado = "Sí" if item.get("vinculado") else ""
            ruta = item.get("ruta", "")
            # Prepare numeric sortable columns: fecha_ts and size_kb
            fecha_ts = None
            try:
                if fecha:
                    # parse YYYY-MM-DD-HHMMSS
                    from datetime import datetime

                    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d-%H%M%S")
                    fecha_ts = fecha_dt.timestamp()
                else:
                    fecha_ts = 0.0
            except Exception:
                fecha_ts = 0.0

            size_kb = human_size_to_kb(size) or -1
            # linked_backup display: prefer basename if it's a path
            linked_raw = item.get("linked_backup") or ""
            linked_display = ""
            try:
                if linked_raw:
                    linked_display = linked_raw.split("/")[-1]
            except Exception:
                linked_display = linked_raw

            # Append row; store the dict in column index 9
            # Columns: icon (0), tipo(1), nombre_markup(2), fecha(3), size(4), estado(5), vinculado(6), ruta(7), bg_color(8), item(9), fecha_ts(10), size_kb(11), linked_backup(12)
            self.liststore.append(
                [
                    icon,
                    tipo,
                    nombre_markup,
                    fecha,
                    size,
                    estado,
                    vinculado,
                    ruta,
                    bg_color,
                    item,
                    float(fecha_ts),
                    int(size_kb),
                    linked_display,
                ]
            )
        # Ensure the TreeView is visible
        self.treeview.show_all()
        self.status_label.set_text(
            f"Total: {len(items)} elementos (snapshots locales y backups externos)"
        )

    # Context menu and activation handlers
    def on_row_activated(self, treeview, path, column):
        model = treeview.get_model()
        treeiter = model.get_iter(path)
        item = model.get_value(treeiter, 9)
        # show info dialog for this item
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Información del elemento",
        )
        dialog.format_secondary_text(
            f"Nombre: {item.get('nombre')}\nTipo: {item.get('type')}\nRuta: {item.get('ruta')}"
        )
        dialog.run()
        dialog.destroy()

    def menu_info_activate(self, menuitem):
        model, treeiter = self.treeview.get_selection().get_selected()
        if not treeiter:
            return
        item = model.get_value(treeiter, 9)
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Información del elemento",
        )
        dialog.format_secondary_text(
            f"Nombre: {item.get('nombre')}\nTipo: {item.get('type')}\nRuta: {item.get('ruta')}"
        )
        dialog.run()
        dialog.destroy()

    def menu_delete_activate(self, menuitem):
        # Reuse on_delete which gets currently selected row
        self.on_delete(None)

    def on_refresh(self, button):
        self.load_unified()

    def on_delete(self, button):
        model, treeiter = self.treeview.get_selection().get_selected()
        if not treeiter:
            self.status_label.set_text("Selecciona un elemento para borrar.")
            return

        item = model.get_value(treeiter, 9)

        self.status_label.set_text(f"Borrando {item['nombre']}...")
        if self.delete_btn:
            self.delete_btn.set_sensitive(False)
        # If item is linked, offer cascade options
        linked = item.get("linked_backup") or ""
        choices = []
        if item.get("type") == "local":
            choices = [
                "Borrar solo snapshot local",
                "Borrar snapshot y backup externo vinculado (si existe)",
                "Cancelar",
            ]
        else:
            choices = [
                "Borrar solo backup externo",
                "Borrar backup y snapshot vinculado (si existe)",
                "Cancelar",
            ]

        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.NONE,
            text=f"¿Qué deseas borrar? {item.get('nombre')}",
        )
        for idx, c in enumerate(choices):
            btn = Gtk.Button(label=c)
            # Map first two choices to responses 1 and 2
            if idx == 0:
                dialog.add_button(c, Gtk.ResponseType.YES)
            elif idx == 1:
                dialog.add_button(c, Gtk.ResponseType.NO)
            else:
                dialog.add_button(c, Gtk.ResponseType.CANCEL)

        resp = dialog.run()
        dialog.destroy()

        if resp == Gtk.ResponseType.CANCEL:
            self.status_label.set_text("Borrado cancelado.")
            if self.delete_btn:
                self.delete_btn.set_sensitive(True)
            return

        do_snapshot = False
        do_backup = False
        if item.get("type") == "local":
            if resp == Gtk.ResponseType.YES:
                do_snapshot = True
            elif resp == Gtk.ResponseType.NO:
                do_snapshot = True
                do_backup = True
        else:
            if resp == Gtk.ResponseType.YES:
                do_backup = True
            elif resp == Gtk.ResponseType.NO:
                do_backup = True
                do_snapshot = True

        # Background deletion using TMManager methods (which will use admin when needed)
        def delete_worker(it_item, do_snap, do_bkp):
            errors = []
            try:
                if do_snap and it_item.get("type") == "local":
                    try:
                        self.manager.delete_snapshot(it_item.get("nombre"))
                        logger.info(f"Deleted snapshot {it_item.get('nombre')}")
                    except Exception as e:
                        errors.append(f"Snapshot: {e}")
                if do_bkp and it_item.get("type") == "externo":
                    try:
                        self.manager.delete_backup(
                            it_item.get("ruta") or it_item.get("nombre")
                        )
                        logger.info(
                            f"Deleted backup {it_item.get('ruta') or it_item.get('nombre')}"
                        )
                    except Exception as e:
                        errors.append(f"Backup: {e}")
                # If we requested to delete both but the selected item is only one type,
                # try to find the linked counterpart and delete it as well
                if (
                    do_bkp
                    and it_item.get("type") == "local"
                    and it_item.get("linked_backup")
                ):
                    try:
                        self.manager.delete_backup(it_item.get("linked_backup"))
                    except Exception as e:
                        errors.append(f"Backup linked: {e}")
                if (
                    do_snap
                    and it_item.get("type") == "externo"
                    and it_item.get("linked_backup")
                ):
                    # for backups, linked_backup stores snapshot name when present
                    try:
                        self.manager.delete_snapshot(it_item.get("linked_backup"))
                    except Exception as e:
                        errors.append(f"Snapshot linked: {e}")
            except Exception as e:
                errors.append(str(e))

            if errors:
                GLib.idle_add(self.on_delete_error, "\n".join(errors))
            else:
                GLib.idle_add(self.on_delete_success, "Eliminación completada")
            GLib.idle_add(self.delete_btn.set_sensitive, True)

        thr = threading.Thread(
            target=delete_worker, args=(item, do_snapshot, do_backup)
        )
        thr.daemon = True
        thr.start()

    def on_delete_all(self, button):
        """Borrar todos los backups externos (solo muestra la advertencia, el usuario debe borrarlos manualmente)."""

        external_backups = []
        mount_point = None
        for row in self.liststore:
            item = row[9]
            if item.get("type") == "externo":
                external_backups.append(item)
                if not mount_point and item.get("ruta"):
                    ruta = item.get("ruta")
                    if ruta.startswith("/Volumes/"):
                        parts = ruta.split("/")
                        if len(parts) >= 3:
                            mount_point = f"/{parts[1]}/{parts[2]}"

        if not external_backups:
            dialog = Gtk.MessageDialog(
                transient_for=self,
                flags=0,
                message_type=Gtk.MessageType.INFO,
                buttons=Gtk.ButtonsType.OK,
                text="No se encontraron backups externos en la lista.",
            )
            dialog.run()
            dialog.destroy()
            return

        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text=f"¿Borrar {len(external_backups)} backups externos?",
        )
        dialog.format_secondary_text(
            "ESTA ES UNA ACCIÓN CRÍTICA. Estás a punto de borrar TODOS los backups externos de Time Machine.\n"
            "Debido a restricciones de permisos de macOS, la aplicación abrirá **Terminal** para que ejecutes el comando.\n\n"
            "¿Estás completamente seguro?"
        )
        response = dialog.run()
        dialog.destroy()

        if response == Gtk.ResponseType.OK:
            self.status_label.set_text(
                "Generando comando de borrado. ¡Revisa la ventana de Terminal!"
            )

            if not mount_point:
                self.on_delete_error(
                    "No se pudo encontrar el punto de montaje de ningún disco de backup externo."
                )
                return

            command = f"sudo tmutil delete -a -d '{mount_point}'"

            full_script = (
                f'tell application "Terminal" to activate\n'
                f'tell application "Terminal" to do script "echo "COPIE Y PEGUE este comando para borrar TODOS los backups: {command}"" in window 1'
            )

            try:
                subprocess.run(
                    ["osascript", "-e", full_script],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.status_label.set_text(
                    "Comando de borrado masivo mostrado en Terminal. Ejecútalo manualmente."
                )
            except Exception as e:
                self.on_delete_error(f"Error al intentar abrir Terminal: {e}")

    def on_delete_success(self, result):
        self.status_label.set_text(f"Elemento borrado. {result}")
        self.load_unified()

    def on_delete_error(self, error_message):
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Error al Borrar",
        )
        dialog.format_secondary_text(
            f"No se pudo borrar el elemento:\n\n{error_message}"
        )
        dialog.run()
        dialog.destroy()
        self.status_label.set_text("Falló el borrado. Actualizando lista...")
        self.load_unified()

    def on_info(self, button):
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Información del sistema y Time Machine",
        )
        info = self.get_system_info()
        dialog.format_secondary_text(info)
        dialog.run()
        dialog.destroy()

    def get_system_info(self):
        try:
            output = subprocess.check_output(["tmutil", "status"], text=True)
            return output
        except Exception as e:
            return f"Error: {e}"

    def on_diag(self, button):
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Diagnóstico rápido",
        )
        info = self.get_diag_info()
        dialog.format_secondary_text(info)
        dialog.run()
        dialog.destroy()

    def get_diag_info(self):
        # Enhanced diagnostics: show python executable in use and do a small `du` test
        import sys

        parts = []
        try:
            parts.append(f"Python executable: {sys.executable}")
            which_proc = subprocess.run(
                ["which", "python3"], capture_output=True, text=True
            )
            parts.append(
                f"which python3: {which_proc.stdout.strip()} (rc={which_proc.returncode})"
            )
        except Exception as e:
            parts.append(f"Failed to discover python path: {e}")

        # Find a candidate path to test `du` on: prefer first external backup from current liststore
        sample_path = None
        try:
            it = self.liststore.get_iter_first()
            while it and sample_path is None:
                item = self.liststore.get_value(it, 9)
                if item and item.get("type") == "externo":
                    sample_path = item.get("ruta")
                    break
                it = self.liststore.iter_next(it)
        except Exception:
            sample_path = None

        if not sample_path:
            # fallback to a generic path (root) which is usually subject to TCC protection
            sample_path = "/"

        parts.append(f"Sample path for du test: {sample_path}")
        # Run local du
        try:
            completed = subprocess.run(
                ["du", "-sk", sample_path], capture_output=True, text=True, timeout=30
            )
            parts.append(f"Local du rc={completed.returncode}")
            if completed.stdout:
                # only include first line of stdout
                parts.append(f"Local du stdout: {completed.stdout.splitlines()[0]}")
            if completed.stderr:
                # include a short snippet of stderr
                snippet = "\n".join(completed.stderr.splitlines()[:5])
                parts.append(f"Local du stderr (snippet): {snippet}")
            logger.debug(
                f"Diagnostic du local returned rc={completed.returncode} stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        except Exception as e:
            parts.append(f"Local du failed: {e}")
            logger.exception(f"Local du test failed for {sample_path}: {e}")

        # Report whether the manager believes it's admin_authorized
        parts.append(
            f"Manager admin_authorized flag: {getattr(self.manager, 'admin_authorized', False)}"
        )

        # Build final output
        return "\n".join(parts)

    def on_help(self, button):
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Ayuda",
        )
        ayuda = """
<b>Time Machine Manager Ultimate</b>

• Lista única de snapshots locales y backups externos
• Borrado seguro (requiere contraseña de admin)
• Botón de información y diagnóstico
• Compatible con macOS 15/16
• Si no ves elementos, revisa permisos de disco y diagnóstico
"""
        dialog.format_secondary_text(ayuda)
        dialog.run()
        dialog.destroy()

    def on_cancel_recalc(self, button):
        # Signal cancellation to the background thread
        self._recalc_cancelled = True
        self.status_label.set_text("Cancelando recalculo...")

    def on_open_full_disk_help(self, button):
        # Show instructions and attempt to open Privacy settings for Full Disk Access
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text="Permisos necesarios: Full Disk Access",
        )
        dialog.format_secondary_text(
            "Para calcular tamaños y acceder a los backups Time Machine, debes conceder 'Full Disk Access' al intérprete Python o a Terminal en Preferencias del Sistema → Privacidad y seguridad → Acceso completo al disco. ¿Abrir Preferencias ahora?"
        )
        resp = dialog.run()
        dialog.destroy()
        if resp == Gtk.ResponseType.OK:
            try:
                # Try to open the Security & Privacy pane to the Privacy section
                subprocess.run(
                    [
                        "open",
                        "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
                    ],
                    check=False,
                )
            except Exception:
                # If that fails, open the general Security & Privacy pane
                try:
                    subprocess.run(
                        ["open", "/System/Applications/System Settings.app"],
                        check=False,
                    )
                except Exception:
                    pass

    def _end_recalc_ui(self):
        # Hide progress UI and re-enable buttons
        self.progress_bar.set_visible(False)
        self.progress_cancel_btn.set_visible(False)
        self.recalc_btn.set_sensitive(True)
        self.request_admin_btn.set_sensitive(True)
        if self._recalc_cancelled:
            self.status_label.set_text("Recalculo cancelado.")
        else:
            self.status_label.set_text("Recalculo completado.")
        return False

    def on_selection_changed(self, selection):
        model, treeiter = selection.get_selected()
        if treeiter:
            ruta = model.get_value(treeiter, 7)
            self.status_label.set_text(ruta)
        else:
            self.status_label.set_text("")

    def on_request_admin(self, button):
        # Ask for admin and update flag
        self.status_label.set_text("Solicitando permisos de administrador...")

        def req():
            ok = self.manager.request_admin()
            GLib.idle_add(self._after_request_admin, ok)

        t = threading.Thread(target=req)
        t.daemon = True
        t.start()

    def _after_request_admin(self, ok):
        if ok:
            self.status_label.set_text(
                "Admin autorizado. Puedes recalcular tamaños ahora."
            )
        else:
            self.status_label.set_text("No se obtuvo autorización admin.")

    def on_recalc_sizes(self, button):
        # Recalculate sizes for all backup rows. Use admin batch if authorized, else attempt non-admin.
        self.status_label.set_text("Recalculando tamaños...")
        self._recalc_cancelled = False
        # show progress UI
        self.progress_bar.set_fraction(0.0)
        self.progress_bar.set_text("Iniciando...")
        self.progress_bar.set_visible(True)
        self.progress_cancel_btn.set_visible(True)
        # disable buttons to avoid concurrent runs
        self.recalc_btn.set_sensitive(False)
        self.request_admin_btn.set_sensitive(False)
        # Build list of backup paths from the model
        paths = []
        rows = []
        for i, row in enumerate(self.liststore):
            item = row[9]
            if item.get("type") == "externo":
                paths.append(item.get("ruta"))
                rows.append((i, item))

        def recalc_thread():
            sizes_map = {}
            try:
                if paths and getattr(self.manager, "admin_authorized", False):
                    # Admin batch: run as admin; show indeterminate progress
                    self._admin_batch = True
                    out_map = self.manager.get_multiple_backup_sizes_admin(paths)
                    sizes_map.update(out_map)
                else:
                    # Non-admin: compute per-path and update progress incrementally
                    total = max(1, len(paths))
                    for idx, p in enumerate(paths):
                        if self._recalc_cancelled:
                            break
                        s = self.manager.get_backup_size(p)
                        sizes_map[p] = s
                        frac = float(idx + 1) / float(total)
                        GLib.idle_add(self.progress_bar.set_fraction, frac)
                        GLib.idle_add(
                            self.progress_bar.set_text,
                            f"{idx+1}/{total} - {p.split('/')[-1]}: {s}",
                        )
            except Exception as e:
                logger.exception(f"Recalc sizes failed: {e}")
            # Update UI rows (unless cancelled)
            if not self._recalc_cancelled:
                GLib.idle_add(self._apply_sizes_map, sizes_map)
            GLib.idle_add(self._end_recalc_ui)

        t = threading.Thread(target=recalc_thread)
        t.daemon = True
        t.start()

    def _apply_sizes_map(self, sizes_map):
        # Update the ListStore entries with new sizes
        changed = 0
        for path, size in sizes_map.items():
            # find rows with this path
            it = self.liststore.get_iter_first()
            while it:
                row_path = self.liststore.get_value(it, 7)
                if row_path == path:
                    item = self.liststore.get_value(it, 9)
                    item["size"] = size
                    # update visible column and numeric KB column for sorting
                    self.liststore.set_value(it, 4, size)
                    kb = human_size_to_kb(size)
                    try:
                        self.liststore.set_value(
                            it, 11, int(kb) if kb is not None else -1
                        )
                    except Exception:
                        pass
                    self.liststore.set_value(it, 9, item)
                    changed += 1
                try:
                    it = self.liststore.iter_next(it)
                except Exception:
                    break
        self.status_label.set_text(f"Recalculado tamaños. Actualizados: {changed}")
        return False

    def on_query_tooltip(self, widget, x, y, keyboard_mode, tooltip):
        # Show ruta for hovered row
        path_info = widget.get_path_at_pos(x, y)
        if not path_info:
            return False
        path, col, cell_x, cell_y = path_info
        model = widget.get_model()
        treeiter = model.get_iter(path)
        ruta = model.get_value(treeiter, 7)
        if ruta:
            tooltip.set_text(ruta)
            return True
        return False

    def on_treeview_button_press(self, treeview, event):
        # Show context menu on right-click
        if event.button == 3:  # right click
            # get row at pointer
            path_info = treeview.get_path_at_pos(int(event.x), int(event.y))
            if path_info:
                path, col, cell_x, cell_y = path_info
                treeview.grab_focus()
                treeview.set_cursor(path, col, False)
                try:
                    # popup expects (parent_menu, parent_menu_shell, func, data, button, activate_time)
                    self.menu.popup(None, None, None, None, event.button, event.time)
                except Exception:
                    try:
                        self.menu.popup(
                            None, None, None, None, 3, Gtk.get_current_event_time()
                        )
                    except Exception:
                        pass
                return True
        return False


# ==== TEST BÁSICO DE TMManager (solo consola) ====
def _test_tmmanager():
    print("--- Test básico de TMManager (no borra nada) ---")
    try:
        tm = TMManager()
        print("--- Solicitando permisos de administrador para el test ---")
        if tm.request_admin():
            print("--- Permisos de administrador concedidos ---")
        else:
            print(
                "--- No se concedieron permisos de administrador, los tamaños pueden no ser correctos ---"
            )
        items = tm.list_snapshots_and_backups()
        print(f"Se encontraron {len(items)} elementos:")
        for it in items:
            print(
                f"- {it['type']}: {it['nombre']} | Fecha: {it['fecha']} | Ruta: {it['ruta']} | Tamaño: {it['size']} | Vinculado: {it.get('vinculado', False)}"
            )
    except Exception as e:
        print(f"ERROR ejecutando TMManager: {e}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        _test_tmmanager()
        sys.exit(0)
    win = TMManagerGUI()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
\n'\n"
                "for p in $list; do\n"
                "  [[ -z \"$p\" ]] && continue\n"
                "  if [[ -e \"$p\" ]]; then\n"
                "    bytes=\"\"\n"
                f"    out=$({TMUTIL} uniquesize \"$p\" 2>/dev/null | head -n1 || true)\n"
                "    if [[ -n \"$out\" ]]; then\n"
                "      bytes=$(echo \"$out\" | LC_ALL=C sed -E 's/[^0-9\.]+//g')\n"
                "      if [[ -z \"$bytes\" ]]; then\n"
                "        kb=$( /usr/bin/du -sk \"$p\" 2>/dev/null | awk '{print $1}' )\n"
                "        bytes=$((kb*1024))\n"
                "      fi\n"
                "    fi\n"
                "    if [[ -z \"$bytes\" ]]; then\n"
                "      kb=$( /usr/bin/du -sk \"$p\" 2>/dev/null | awk '{print $1}' )\n"
                "      bytes=$((kb*1024))\n"
                "    fi\n"
                "    echo -e \"$p\t$bytes\"\n"
                "  fi\n"
                "done\n"
            )
            out = self._run_as_admin_command(["/bin/bash", "-lc", script])
            result = {}
            for line in (out or "").splitlines():
                if not line.strip():
                    continue
                try:
                    path_part, bytes_part = line.split("\t", 1)
                    num = re.sub(r"[^0-9\.]+", "", bytes_part.strip())
                    bytes_val = float(num) if num else 0.0
                    result[path_part] = format_bytes(bytes_val)
                except Exception:
                    continue
            return result
        except Exception as e:
            logger.exception(f"Failed to list backups and sizes via batch admin: {e}")
            return {}

    def resolve_backup_path(self, path):
        """Resolve a backup identifier (either full path or basename) to a full backup path.

        Strategy:
        - If path contains '/', assume it's already a full path and return it.
        - Otherwise, call `tmutil listbackups` and try matches in order:
          1) basename equality
          2) line endswith(path)
          3) line contains path
        - For any candidate, prefer the one where os.path.exists(candidate) is True.
        - Return the first good candidate, or None if none found.
        """
        import os

        try:
            if "/" in (path or ""):
                return path
            try:
                # Use admin to ensure we can see all backups to resolve the path
                backups_out = self.run_tmutil(["listbackups"], admin=True)
            except Exception as e:
                logger.debug(f"Could not list backups to resolve {path}: {e}")
                return None

            candidates = []
            for line in backups_out.splitlines():
                line = line.strip()
                if not line:
                    continue
                # basename equality
                try:
                    if os.path.basename(line) == path:
                        candidates.insert(0, line)
                        continue
                except Exception:
                    pass
                # endswith
                if line.endswith(path):
                    candidates.append(line)
                    continue
                # contains
                if path in line:
                    candidates.append(line)

            # Prefer existing paths
            for c in candidates:
                try:
                    if os.path.exists(c):
                        return c
                except Exception:
                    pass

            # fallback to first candidate if any
            if candidates:
                return candidates[0]
        except Exception:
            logger.exception(f"Error resolving backup path for {path}")
        return None

    def _find_previous_for_backup(self, marker_path):
        """Given a path like /Volumes/.timemachine/<UUID>/<name>.backup, try to
        find the actual '.previous' directory on mounted backup volumes.

        Strategy:
        - Extract the basename (e.g. 2025-10-01-141748.backup -> 2025-10-01-141748)
        - Look under each mount point in /Volumes (excluding .timemachine) for a
          directory named <basename>.previous and return the first existing path.
        - Return None if not found.
        """
        import os

        try:
            base = os.path.basename(marker_path)
            if base.endswith(".backup"):
                name = base.replace(".backup", "")
            else:
                name = base
            candidate_name = f"{name}.previous"
            for entry in sorted(os.listdir("/Volumes")):
                if entry == ".timemachine":
                    continue
                mountp = os.path.join("/Volumes", entry)
                try:
                    cand = os.path.join(mountp, candidate_name)
                    if os.path.exists(cand):
                        return cand
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def delete_snapshot(self, date):
        logger.info(f"Deleting local snapshot: {date}")
        return self.run_tmutil(["deletelocalsnapshots", date], admin=True)

    def delete_backup(self, path):
        logger.info(f"Deleting backup: {path}")
        try:
            resolved = self.resolve_backup_path(path)
            if resolved:
                logger.debug(f"Resolved backup {path} -> {resolved}")
                path = resolved
            else:
                logger.debug(
                    f"Could not resolve backup path for {path}; proceeding with original value"
                )
        except Exception:
            logger.exception(f"Error while resolving backup path for deletion: {path}")

        return self.run_tmutil(["delete", path], admin=True)


class TMManagerGUI(Gtk.Window):
    def __init__(self):
        super().__init__(title="Time Machine Manager Ultimate DEFINITIVO")
        self.set_default_size(1000, 700)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.manager = TMManager()
        # Run a startup TCC check to detect if the running Python lacks Full Disk Access
        self.create_main_ui()
        # Ask for admin privileges after the main UI is shown so the dialog is visible
        # and not blocking the constructor. This schedules a one-shot call on the
        # main loop to show the prompt once the window is visible.
        GLib.idle_add(self._ask_for_admin_dialog)
        # perform the check after UI is ready so we can show dialogs
        self._startup_tcc_check()

    def _ask_for_admin_dialog(self):
        """Show the initial admin request dialog once the UI is ready.

        Returned False to run only once when scheduled with GLib.idle_add.
        """
        try:
            dialog = Gtk.MessageDialog(
                parent=self,
                flags=0,
                message_type=Gtk.MessageType.QUESTION,
                buttons=Gtk.ButtonsType.YES_NO,
                text="Pedir permisos de administrador",
            )
            dialog.format_secondary_text(
                "¿Deseas conceder permisos de administrador para que la app pueda calcular tamaños y borrar backups sin restricciones? (Se pedirá la contraseña)"
            )
            response = dialog.run()
            dialog.destroy()
            if response == Gtk.ResponseType.YES:
                ok = self.manager.request_admin()
                self.manager.admin_authorized = bool(ok)
        except Exception:
            # If anything goes wrong showing the dialog, log and continue silently
            logger.exception("Failed to show initial admin dialog")
        return False

    def _startup_tcc_check(self):
        import shlex
        import sys

        def check_thread():
            # pick a sample path: prefer first external backup if available
            sample = None
            try:
                it = self.liststore.get_iter_first()
                while it:
                    item = self.liststore.get_value(it, 9)
                    if item and item.get("type") == "externo":
                        sample = item.get("ruta")
                        break
                    it = self.liststore.iter_next(it)
            except Exception:
                sample = None
            if not sample:
                sample = "/"

            logger.debug(
                f"Startup TCC check: testing ls on {sample} using {sys.executable}"
            )
            try:
                completed = subprocess.run(
                    ["ls", "-l", sample], capture_output=True, text=True, timeout=20
                )
                logger.debug(
                    f"Startup ls rc={completed.returncode} stdout={completed.stdout!r} stderr={completed.stderr!r}"
                )
                # Consider Operation not permitted or non-zero return code with stderr as failure
                if completed.returncode != 0 or (
                    "Operation not permitted" in (completed.stderr or "")
                    or "Permission denied" in (completed.stderr or "")
                ):
                    # Show dialog on main thread
                    GLib.idle_add(
                        self._show_tcc_dialog,
                        sys.executable,
                        completed.returncode,
                        completed.stderr,
                    )
            except Exception as e:
                logger.exception(f"Startup ls test failed: {e}")
                GLib.idle_add(self._show_tcc_dialog, sys.executable, -1, str(e))

        t = threading.Thread(target=check_thread)
        t.daemon = True
        t.start()

    def _show_tcc_dialog(self, py_executable, rc, stderr_text):
        # Inform the user that Full Disk Access may be needed and show actionable steps
        msg = f"El intérprete Python en uso es:\n{py_executable}\n\nResultado del test de permisos: rc={rc}\n\nErrores:\n{(stderr_text or '')[:1000]}\n\nSi ves 'Operation not permitted' o 'Permission denied', añade el ejecutable anterior a Preferencias → Privacidad y seguridad → Acceso completo al disco y reinicia la aplicación. ¿Abrir Preferencias ahora?"
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text="Full Disk Access probablemente requerido",
        )
        dialog.format_secondary_text(msg)
        resp = dialog.run()
        dialog.destroy()
        if resp == Gtk.ResponseType.OK:
            try:
                subprocess.run(
                    [
                        "open",
                        "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
                    ],
                    check=False,
                )
            except Exception:
                try:
                    subprocess.run(
                        ["open", "/System/Applications/System Settings.app"],
                        check=False,
                    )
                except Exception:
                    pass
        return False

    def create_main_ui(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        refresh_btn = Gtk.Button(label="Refrescar")
        refresh_btn.connect("clicked", self.on_refresh)
        button_box.pack_start(refresh_btn, False, False, 0)
        info_btn = Gtk.Button(label="Información")
        info_btn.connect("clicked", self.on_info)
        button_box.pack_start(info_btn, False, False, 0)
        diag_btn = Gtk.Button(label="Diagnóstico")
        diag_btn.connect("clicked", self.on_diag)
        button_box.pack_start(diag_btn, False, False, 0)
        help_btn = Gtk.Button(label="Ayuda")
        help_btn.connect("clicked", self.on_help)
        button_box.pack_start(help_btn, False, False, 0)
        # Request admin button (user can ask for admin later)
        self.request_admin_btn = Gtk.Button(label="Pedir permisos admin")
        self.request_admin_btn.connect("clicked", self.on_request_admin)
        button_box.pack_start(self.request_admin_btn, False, False, 0)
        # Full Disk Access help button
        fda_btn = Gtk.Button(label="Permisos de disco (Full Disk Access)")
        fda_btn.connect("clicked", self.on_open_full_disk_help)
        button_box.pack_start(fda_btn, False, False, 0)
        # Recalculate sizes button (uses admin batch if available)
        self.recalc_btn = Gtk.Button(label="Recalcular tamaños")
        self.recalc_btn.connect("clicked", self.on_recalc_sizes)
        button_box.pack_start(self.recalc_btn, False, False, 0)
        self.delete_btn = Gtk.Button(label="Borrar seleccionado")
        self.delete_btn.connect("clicked", self.on_delete)
        button_box.pack_end(self.delete_btn, False, False, 0)
        # Delete all button
        self.delete_all_btn = Gtk.Button(label="Borrar todo")
        self.delete_all_btn.connect("clicked", self.on_delete_all)
        button_box.pack_end(self.delete_all_btn, False, False, 0)
        # Exit button
        exit_btn = Gtk.Button(label="Salir")
        exit_btn.connect("clicked", lambda b: Gtk.main_quit())
        button_box.pack_end(exit_btn, False, False, 0)
        vbox.pack_start(button_box, False, False, 0)
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(500)
        # Use a TreeView with a ListStore to display a formatted table
        # Columns: Icon, Tipo, Nombre (markup), Fecha, Tamaño, Estado, Vinculado, Ruta (hidden), bg_color, item(dict)
        # Additional hidden columns for sorting and linked backup:
        # fecha_ts (float) used to sort by date, size_kb (int) used to sort by size, linked_backup (str)
        # bg_color is used to set per-row background (e.g. linked or backup rows)
        self.liststore = Gtk.ListStore(
            str, str, str, str, str, str, str, str, str, object, float, int, str
        )
        self.treeview = Gtk.TreeView(model=self.liststore)
        self.treeview.set_activate_on_single_click(True)

        def add_column(title, col_id, use_markup=False, visible=True, set_bg=False):
            renderer = Gtk.CellRendererText()
            if use_markup:
                column = Gtk.TreeViewColumn(title, renderer)
                column.add_attribute(renderer, "markup", col_id)
            else:
                column = Gtk.TreeViewColumn(title, renderer, text=col_id)
            if set_bg:
                # bind the cell background to the bg_color column (index 8)
                column.add_attribute(renderer, "cell-background", 8)
            column.set_visible(visible)
            column.set_resizable(True)
            self.treeview.append_column(column)
            return column

        # Icon renderer (simple letter)
        icon_renderer = Gtk.CellRendererText()
        icon_renderer.props.xalign = 0.5
        icon_column = Gtk.TreeViewColumn("", icon_renderer, text=0)
        icon_column.set_resizable(False)
        icon_column.set_min_width(48)
        self.treeview.append_column(icon_column)

        add_column("Tipo", 1)
        # Name column: use larger markup for better readability
        add_column("Nombre", 2, use_markup=True, set_bg=True)
        fecha_col = add_column("Fecha", 3, set_bg=True)
        size_col = add_column("Tamaño", 4, set_bg=True)
        add_column("Estado", 5, set_bg=True)
        add_column("Vinculado", 6, set_bg=True)
        # Ruta (oculta por defecto, pero disponible en el modelo)
        add_column("Ruta", 7, visible=False)
        # Linked backup (show basename or short path)
        linked_col = add_column("Backup vinculado", 12, visible=True)

        # Enable sorting: Fecha by fecha_ts (index 10), Tamaño by size_kb (index 11)
        try:
            fecha_col.set_sort_column_id(10)
            size_col.set_sort_column_id(11)
            # also mark linked_col sortable by its text value (index 12)
            linked_col.set_sort_column_id(12)
        except Exception:
            # if something goes wrong, continue without raising
            logger.exception("Could not set sort columns")

        # Selection handler to show ruta in status bar
        selection = self.treeview.get_selection()
        selection.connect("changed", self.on_selection_changed)

        # Enable query tooltip for showing ruta on hover
        self.treeview.set_has_tooltip(True)
        self.treeview.connect("query-tooltip", self.on_query_tooltip)

        scrolled.add(self.treeview)
        vbox.pack_start(scrolled, True, True, 0)
        self.status_label = Gtk.Label()
        self.status_label.set_halign(Gtk.Align.START)
        vbox.pack_start(self.status_label, False, False, 0)
        # Progress area for recalculating sizes
        progress_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_show_text(True)
        self.progress_bar.set_visible(False)
        progress_box.pack_start(self.progress_bar, True, True, 0)
        self.progress_cancel_btn = Gtk.Button(label="Cancelar")
        self.progress_cancel_btn.connect("clicked", self.on_cancel_recalc)
        self.progress_cancel_btn.set_visible(False)
        progress_box.pack_start(self.progress_cancel_btn, False, False, 0)
        vbox.pack_start(progress_box, False, False, 0)
        self._recalc_cancelled = False
        # Context menu for rows (right click)
        self.menu = Gtk.Menu()
        mi_info = Gtk.MenuItem(label="Información")
        mi_info.connect("activate", self.menu_info_activate)
        self.menu.append(mi_info)
        mi_delete = Gtk.MenuItem(label="Borrar")
        mi_delete.connect("activate", self.menu_delete_activate)
        self.menu.append(mi_delete)
        self.menu.show_all()

        # Double-click (row-activated) to show info
        self.treeview.connect("row-activated", self.on_row_activated)
        # Right click to show context menu
        self.treeview.connect("button-press-event", self.on_treeview_button_press)
        self.add(vbox)
        self.load_unified()

    def load_unified(self):
        # Clear liststore
        try:
            self.liststore.clear()
        except Exception:
            pass
        self.status_label.set_text("Cargando lista...")

        def load_thread():
            items = self.manager.list_snapshots_and_backups()
            GLib.idle_add(self.update_unified_list, items)

        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()

    def update_unified_list(self, items):
        if not items:
            self.status_label.set_text("No se encontraron snapshots ni backups")
            return

        # Populate liststore with formatted values. Use markup for Nombre to highlight it.
        for item in items:
            icon = "L" if item["type"] == "local" else "E"
            tipo = "Local" if item["type"] == "local" else "Externo"
            # Highlight linked items with a colored name
            if item.get("vinculado"):
                nombre_markup = (
                    f"<span foreground=\"#008000\"><b>{item['nombre']}</b></span>"
                )
                bg_color = "#f0fff0"  # light green
            else:
                nombre_markup = f"<b>{item['nombre']}</b>"
                bg_color = "#ffffff"
            fecha = item.get("fecha", "")
            size = item.get("size", "")
            estado = item.get("estado", "")
            vinculado = "Sí" if item.get("vinculado") else ""
            ruta = item.get("ruta", "")
            # Prepare numeric sortable columns: fecha_ts and size_kb
            fecha_ts = None
            try:
                if fecha:
                    # parse YYYY-MM-DD-HHMMSS
                    from datetime import datetime

                    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d-%H%M%S")
                    fecha_ts = fecha_dt.timestamp()
                else:
                    fecha_ts = 0.0
            except Exception:
                fecha_ts = 0.0

            size_kb = human_size_to_kb(size) or -1
            # linked_backup display: prefer basename if it's a path
            linked_raw = item.get("linked_backup") or ""
            linked_display = ""
            try:
                if linked_raw:
                    linked_display = linked_raw.split("/")[-1]
            except Exception:
                linked_display = linked_raw

            # Append row; store the dict in column index 9
            # Columns: icon (0), tipo(1), nombre_markup(2), fecha(3), size(4), estado(5), vinculado(6), ruta(7), bg_color(8), item(9), fecha_ts(10), size_kb(11), linked_backup(12)
            self.liststore.append(
                [
                    icon,
                    tipo,
                    nombre_markup,
                    fecha,
                    size,
                    estado,
                    vinculado,
                    ruta,
                    bg_color,
                    item,
                    float(fecha_ts),
                    int(size_kb),
                    linked_display,
                ]
            )
        # Ensure the TreeView is visible
        self.treeview.show_all()
        self.status_label.set_text(
            f"Total: {len(items)} elementos (snapshots locales y backups externos)"
        )

    # Context menu and activation handlers
    def on_row_activated(self, treeview, path, column):
        model = treeview.get_model()
        treeiter = model.get_iter(path)
        item = model.get_value(treeiter, 9)
        # show info dialog for this item
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Información del elemento",
        )
        dialog.format_secondary_text(
            f"Nombre: {item.get('nombre')}\nTipo: {item.get('type')}\nRuta: {item.get('ruta')}"
        )
        dialog.run()
        dialog.destroy()

    def menu_info_activate(self, menuitem):
        model, treeiter = self.treeview.get_selection().get_selected()
        if not treeiter:
            return
        item = model.get_value(treeiter, 9)
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Información del elemento",
        )
        dialog.format_secondary_text(
            f"Nombre: {item.get('nombre')}\nTipo: {item.get('type')}\nRuta: {item.get('ruta')}"
        )
        dialog.run()
        dialog.destroy()

    def menu_delete_activate(self, menuitem):
        # Reuse on_delete which gets currently selected row
        self.on_delete(None)

    def on_refresh(self, button):
        self.load_unified()

    def on_delete(self, button):
        model, treeiter = self.treeview.get_selection().get_selected()
        if not treeiter:
            self.status_label.set_text("Selecciona un elemento para borrar.")
            return

        item = model.get_value(treeiter, 9)

        self.status_label.set_text(f"Borrando {item['nombre']}...")
        if self.delete_btn:
            self.delete_btn.set_sensitive(False)
        # If item is linked, offer cascade options
        linked = item.get("linked_backup") or ""
        choices = []
        if item.get("type") == "local":
            choices = [
                "Borrar solo snapshot local",
                "Borrar snapshot y backup externo vinculado (si existe)",
                "Cancelar",
            ]
        else:
            choices = [
                "Borrar solo backup externo",
                "Borrar backup y snapshot vinculado (si existe)",
                "Cancelar",
            ]

        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.NONE,
            text=f"¿Qué deseas borrar? {item.get('nombre')}",
        )
        for idx, c in enumerate(choices):
            btn = Gtk.Button(label=c)
            # Map first two choices to responses 1 and 2
            if idx == 0:
                dialog.add_button(c, Gtk.ResponseType.YES)
            elif idx == 1:
                dialog.add_button(c, Gtk.ResponseType.NO)
            else:
                dialog.add_button(c, Gtk.ResponseType.CANCEL)

        resp = dialog.run()
        dialog.destroy()

        if resp == Gtk.ResponseType.CANCEL:
            self.status_label.set_text("Borrado cancelado.")
            if self.delete_btn:
                self.delete_btn.set_sensitive(True)
            return

        do_snapshot = False
        do_backup = False
        if item.get("type") == "local":
            if resp == Gtk.ResponseType.YES:
                do_snapshot = True
            elif resp == Gtk.ResponseType.NO:
                do_snapshot = True
                do_backup = True
        else:
            if resp == Gtk.ResponseType.YES:
                do_backup = True
            elif resp == Gtk.ResponseType.NO:
                do_backup = True
                do_snapshot = True

        # Background deletion using TMManager methods (which will use admin when needed)
        def delete_worker(it_item, do_snap, do_bkp):
            errors = []
            try:
                if do_snap and it_item.get("type") == "local":
                    try:
                        self.manager.delete_snapshot(it_item.get("nombre"))
                        logger.info(f"Deleted snapshot {it_item.get('nombre')}")
                    except Exception as e:
                        errors.append(f"Snapshot: {e}")
                if do_bkp and it_item.get("type") == "externo":
                    try:
                        self.manager.delete_backup(
                            it_item.get("ruta") or it_item.get("nombre")
                        )
                        logger.info(
                            f"Deleted backup {it_item.get('ruta') or it_item.get('nombre')}"
                        )
                    except Exception as e:
                        errors.append(f"Backup: {e}")
                # If we requested to delete both but the selected item is only one type,
                # try to find the linked counterpart and delete it as well
                if (
                    do_bkp
                    and it_item.get("type") == "local"
                    and it_item.get("linked_backup")
                ):
                    try:
                        self.manager.delete_backup(it_item.get("linked_backup"))
                    except Exception as e:
                        errors.append(f"Backup linked: {e}")
                if (
                    do_snap
                    and it_item.get("type") == "externo"
                    and it_item.get("linked_backup")
                ):
                    # for backups, linked_backup stores snapshot name when present
                    try:
                        self.manager.delete_snapshot(it_item.get("linked_backup"))
                    except Exception as e:
                        errors.append(f"Snapshot linked: {e}")
            except Exception as e:
                errors.append(str(e))

            if errors:
                GLib.idle_add(self.on_delete_error, "\n".join(errors))
            else:
                GLib.idle_add(self.on_delete_success, "Eliminación completada")
            GLib.idle_add(self.delete_btn.set_sensitive, True)

        thr = threading.Thread(
            target=delete_worker, args=(item, do_snapshot, do_backup)
        )
        thr.daemon = True
        thr.start()

    def on_delete_all(self, button):
        """Borrar todos los backups externos (solo muestra la advertencia, el usuario debe borrarlos manualmente)."""

        external_backups = []
        mount_point = None
        for row in self.liststore:
            item = row[9]
            if item.get("type") == "externo":
                external_backups.append(item)
                if not mount_point and item.get("ruta"):
                    ruta = item.get("ruta")
                    if ruta.startswith("/Volumes/"):
                        parts = ruta.split("/")
                        if len(parts) >= 3:
                            mount_point = f"/{parts[1]}/{parts[2]}"

        if not external_backups:
            dialog = Gtk.MessageDialog(
                transient_for=self,
                flags=0,
                message_type=Gtk.MessageType.INFO,
                buttons=Gtk.ButtonsType.OK,
                text="No se encontraron backups externos en la lista.",
            )
            dialog.run()
            dialog.destroy()
            return

        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text=f"¿Borrar {len(external_backups)} backups externos?",
        )
        dialog.format_secondary_text(
            "ESTA ES UNA ACCIÓN CRÍTICA. Estás a punto de borrar TODOS los backups externos de Time Machine.\n"
            "Debido a restricciones de permisos de macOS, la aplicación abrirá **Terminal** para que ejecutes el comando.\n\n"
            "¿Estás completamente seguro?"
        )
        response = dialog.run()
        dialog.destroy()

        if response == Gtk.ResponseType.OK:
            self.status_label.set_text(
                "Generando comando de borrado. ¡Revisa la ventana de Terminal!"
            )

            if not mount_point:
                self.on_delete_error(
                    "No se pudo encontrar el punto de montaje de ningún disco de backup externo."
                )
                return

            command = f"sudo tmutil delete -a -d '{mount_point}'"

            full_script = (
                f'tell application "Terminal" to activate\n'
                f'tell application "Terminal" to do script "echo "COPIE Y PEGUE este comando para borrar TODOS los backups: {command}"" in window 1'
            )

            try:
                subprocess.run(
                    ["osascript", "-e", full_script],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.status_label.set_text(
                    "Comando de borrado masivo mostrado en Terminal. Ejecútalo manualmente."
                )
            except Exception as e:
                self.on_delete_error(f"Error al intentar abrir Terminal: {e}")

    def on_delete_success(self, result):
        self.status_label.set_text(f"Elemento borrado. {result}")
        self.load_unified()

    def on_delete_error(self, error_message):
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Error al Borrar",
        )
        dialog.format_secondary_text(
            f"No se pudo borrar el elemento:\n\n{error_message}"
        )
        dialog.run()
        dialog.destroy()
        self.status_label.set_text("Falló el borrado. Actualizando lista...")
        self.load_unified()

    def on_info(self, button):
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Información del sistema y Time Machine",
        )
        info = self.get_system_info()
        dialog.format_secondary_text(info)
        dialog.run()
        dialog.destroy()

    def get_system_info(self):
        try:
            output = subprocess.check_output(["tmutil", "status"], text=True)
            return output
        except Exception as e:
            return f"Error: {e}"

    def on_diag(self, button):
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Diagnóstico rápido",
        )
        info = self.get_diag_info()
        dialog.format_secondary_text(info)
        dialog.run()
        dialog.destroy()

    def get_diag_info(self):
        # Enhanced diagnostics: show python executable in use and do a small `du` test
        import sys

        parts = []
        try:
            parts.append(f"Python executable: {sys.executable}")
            which_proc = subprocess.run(
                ["which", "python3"], capture_output=True, text=True
            )
            parts.append(
                f"which python3: {which_proc.stdout.strip()} (rc={which_proc.returncode})"
            )
        except Exception as e:
            parts.append(f"Failed to discover python path: {e}")

        # Find a candidate path to test `du` on: prefer first external backup from current liststore
        sample_path = None
        try:
            it = self.liststore.get_iter_first()
            while it and sample_path is None:
                item = self.liststore.get_value(it, 9)
                if item and item.get("type") == "externo":
                    sample_path = item.get("ruta")
                    break
                it = self.liststore.iter_next(it)
        except Exception:
            sample_path = None

        if not sample_path:
            # fallback to a generic path (root) which is usually subject to TCC protection
            sample_path = "/"

        parts.append(f"Sample path for du test: {sample_path}")
        # Run local du
        try:
            completed = subprocess.run(
                ["du", "-sk", sample_path], capture_output=True, text=True, timeout=30
            )
            parts.append(f"Local du rc={completed.returncode}")
            if completed.stdout:
                # only include first line of stdout
                parts.append(f"Local du stdout: {completed.stdout.splitlines()[0]}")
            if completed.stderr:
                # include a short snippet of stderr
                snippet = "\n".join(completed.stderr.splitlines()[:5])
                parts.append(f"Local du stderr (snippet): {snippet}")
            logger.debug(
                f"Diagnostic du local returned rc={completed.returncode} stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        except Exception as e:
            parts.append(f"Local du failed: {e}")
            logger.exception(f"Local du test failed for {sample_path}: {e}")

        # Report whether the manager believes it's admin_authorized
        parts.append(
            f"Manager admin_authorized flag: {getattr(self.manager, 'admin_authorized', False)}"
        )

        # Build final output
        return "\n".join(parts)

    def on_help(self, button):
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Ayuda",
        )
        ayuda = """
<b>Time Machine Manager Ultimate</b>

• Lista única de snapshots locales y backups externos
• Borrado seguro (requiere contraseña de admin)
• Botón de información y diagnóstico
• Compatible con macOS 15/16
• Si no ves elementos, revisa permisos de disco y diagnóstico
"""
        dialog.format_secondary_text(ayuda)
        dialog.run()
        dialog.destroy()

    def on_cancel_recalc(self, button):
        # Signal cancellation to the background thread
        self._recalc_cancelled = True
        self.status_label.set_text("Cancelando recalculo...")

    def on_open_full_disk_help(self, button):
        # Show instructions and attempt to open Privacy settings for Full Disk Access
        dialog = Gtk.MessageDialog(
            parent=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text="Permisos necesarios: Full Disk Access",
        )
        dialog.format_secondary_text(
            "Para calcular tamaños y acceder a los backups Time Machine, debes conceder 'Full Disk Access' al intérprete Python o a Terminal en Preferencias del Sistema → Privacidad y seguridad → Acceso completo al disco. ¿Abrir Preferencias ahora?"
        )
        resp = dialog.run()
        dialog.destroy()
        if resp == Gtk.ResponseType.OK:
            try:
                # Try to open the Security & Privacy pane to the Privacy section
                subprocess.run(
                    [
                        "open",
                        "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
                    ],
                    check=False,
                )
            except Exception:
                # If that fails, open the general Security & Privacy pane
                try:
                    subprocess.run(
                        ["open", "/System/Applications/System Settings.app"],
                        check=False,
                    )
                except Exception:
                    pass

    def _end_recalc_ui(self):
        # Hide progress UI and re-enable buttons
        self.progress_bar.set_visible(False)
        self.progress_cancel_btn.set_visible(False)
        self.recalc_btn.set_sensitive(True)
        self.request_admin_btn.set_sensitive(True)
        if self._recalc_cancelled:
            self.status_label.set_text("Recalculo cancelado.")
        else:
            self.status_label.set_text("Recalculo completado.")
        return False

    def on_selection_changed(self, selection):
        model, treeiter = selection.get_selected()
        if treeiter:
            ruta = model.get_value(treeiter, 7)
            self.status_label.set_text(ruta)
        else:
            self.status_label.set_text("")

    def on_request_admin(self, button):
        # Ask for admin and update flag
        self.status_label.set_text("Solicitando permisos de administrador...")

        def req():
            ok = self.manager.request_admin()
            GLib.idle_add(self._after_request_admin, ok)

        t = threading.Thread(target=req)
        t.daemon = True
        t.start()

    def _after_request_admin(self, ok):
        if ok:
            self.status_label.set_text(
                "Admin autorizado. Puedes recalcular tamaños ahora."
            )
        else:
            self.status_label.set_text("No se obtuvo autorización admin.")

    def on_recalc_sizes(self, button):
        # Recalculate sizes for all backup rows. Use admin batch if authorized, else attempt non-admin.
        self.status_label.set_text("Recalculando tamaños...")
        self._recalc_cancelled = False
        # show progress UI
        self.progress_bar.set_fraction(0.0)
        self.progress_bar.set_text("Iniciando...")
        self.progress_bar.set_visible(True)
        self.progress_cancel_btn.set_visible(True)
        # disable buttons to avoid concurrent runs
        self.recalc_btn.set_sensitive(False)
        self.request_admin_btn.set_sensitive(False)
        # Build list of backup paths from the model
        paths = []
        rows = []
        for i, row in enumerate(self.liststore):
            item = row[9]
            if item.get("type") == "externo":
                paths.append(item.get("ruta"))
                rows.append((i, item))

        def recalc_thread():
            sizes_map = {}
            try:
                if paths and getattr(self.manager, "admin_authorized", False):
                    # Admin batch: run as admin; show indeterminate progress
                    self._admin_batch = True
                    out_map = self.manager.get_multiple_backup_sizes_admin(paths)
                    sizes_map.update(out_map)
                else:
                    # Non-admin: compute per-path and update progress incrementally
                    total = max(1, len(paths))
                    for idx, p in enumerate(paths):
                        if self._recalc_cancelled:
                            break
                        s = self.manager.get_backup_size(p)
                        sizes_map[p] = s
                        frac = float(idx + 1) / float(total)
                        GLib.idle_add(self.progress_bar.set_fraction, frac)
                        GLib.idle_add(
                            self.progress_bar.set_text,
                            f"{idx+1}/{total} - {p.split('/')[-1]}: {s}",
                        )
            except Exception as e:
                logger.exception(f"Recalc sizes failed: {e}")
            # Update UI rows (unless cancelled)
            if not self._recalc_cancelled:
                GLib.idle_add(self._apply_sizes_map, sizes_map)
            GLib.idle_add(self._end_recalc_ui)

        t = threading.Thread(target=recalc_thread)
        t.daemon = True
        t.start()

    def _apply_sizes_map(self, sizes_map):
        # Update the ListStore entries with new sizes
        changed = 0
        for path, size in sizes_map.items():
            # find rows with this path
            it = self.liststore.get_iter_first()
            while it:
                row_path = self.liststore.get_value(it, 7)
                if row_path == path:
                    item = self.liststore.get_value(it, 9)
                    item["size"] = size
                    # update visible column and numeric KB column for sorting
                    self.liststore.set_value(it, 4, size)
                    kb = human_size_to_kb(size)
                    try:
                        self.liststore.set_value(
                            it, 11, int(kb) if kb is not None else -1
                        )
                    except Exception:
                        pass
                    self.liststore.set_value(it, 9, item)
                    changed += 1
                try:
                    it = self.liststore.iter_next(it)
                except Exception:
                    break
        self.status_label.set_text(f"Recalculado tamaños. Actualizados: {changed}")
        return False

    def on_query_tooltip(self, widget, x, y, keyboard_mode, tooltip):
        # Show ruta for hovered row
        path_info = widget.get_path_at_pos(x, y)
        if not path_info:
            return False
        path, col, cell_x, cell_y = path_info
        model = widget.get_model()
        treeiter = model.get_iter(path)
        ruta = model.get_value(treeiter, 7)
        if ruta:
            tooltip.set_text(ruta)
            return True
        return False

    def on_treeview_button_press(self, treeview, event):
        # Show context menu on right-click
        if event.button == 3:  # right click
            # get row at pointer
            path_info = treeview.get_path_at_pos(int(event.x), int(event.y))
            if path_info:
                path, col, cell_x, cell_y = path_info
                treeview.grab_focus()
                treeview.set_cursor(path, col, False)
                try:
                    # popup expects (parent_menu, parent_menu_shell, func, data, button, activate_time)
                    self.menu.popup(None, None, None, None, event.button, event.time)
                except Exception:
                    try:
                        self.menu.popup(
                            None, None, None, None, 3, Gtk.get_current_event_time()
                        )
                    except Exception:
                        pass
                return True
        return False


# ==== TEST BÁSICO DE TMManager (solo consola) ====
def _test_tmmanager():
    print("--- Test básico de TMManager (no borra nada) ---")
    try:
        tm = TMManager()
        print("--- Solicitando permisos de administrador para el test ---")
        if tm.request_admin():
            print("--- Permisos de administrador concedidos ---")
        else:
            print(
                "--- No se concedieron permisos de administrador, los tamaños pueden no ser correctos ---"
            )
        items = tm.list_snapshots_and_backups()
        print(f"Se encontraron {len(items)} elementos:")
        for it in items:
            print(
                f"- {it['type']}: {it['nombre']} | Fecha: {it['fecha']} | Ruta: {it['ruta']} | Tamaño: {it['size']} | Vinculado: {it.get('vinculado', False)}"
            )
    except Exception as e:
        print(f"ERROR ejecutando TMManager: {e}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        _test_tmmanager()
        sys.exit(0)
    win = TMManagerGUI()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
