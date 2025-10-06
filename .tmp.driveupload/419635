
import os
import subprocess
from setuptools import setup

# --- Helper to find GObject introspection data ---
def get_gi_data_files():
    """
    Finds GObject introspection data files (.typelib)
    and returns them in a format suitable for py2app's data_files.
    """
    try:
        # Use pkg-config to find the base directory for typelib files
        command = ['pkg-config', '--variable=typelibdir', 'gobject-introspection-1.0']
        typelib_path = subprocess.check_output(command, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback for when pkg-config is not available or fails
        # This path is common on Homebrew
        homebrew_prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
        typelib_path = os.path.join(homebrew_prefix, 'lib', 'girepository-1.0')
        if not os.path.isdir(typelib_path):
            # A final fallback
            typelib_path = '/usr/local/lib/girepository-1.0'

    if not os.path.isdir(typelib_path):
        print(f"Warning: GObject typelib directory not found at {typelib_path}")
        return []

    data_files = []
    for f in os.listdir(typelib_path):
        if f.endswith('.typelib'):
            data_files.append(os.path.join(typelib_path, f))
    
    # Return in the format py2app expects: [('destination_dir', [source_files])]
    return [('lib/girepository-1.0', data_files)]

# --- py2app configuration ---
APP = ['tm-manager-ultimate-definitivo.py']

# Include GObject introspection data
DATA_FILES = get_gi_data_files()

OPTIONS = {
    'argv_emulation': True,
    'packages': [
        'gi',
        'cairo',
    ],
    'includes': [
        'gi.repository.Gtk',
        'gi.repository.GLib',
        'gi.repository.GObject',
        'gi.repository.Gdk',
        'gi.repository.GdkPixbuf',
        'gi.repository.Pango',
        'shlex',
    ],
    # Ensure py2app knows where to find GTK libraries if they are not in standard paths
    # This might be needed on some systems.
    'frameworks': [],
    'plist': {
        'CFBundleName': 'TM Manager Ultimate',
        'CFBundleDisplayName': 'TM Manager Ultimate',
        'CFBundleGetInfoString': "Time Machine Backup Manager",
        'CFBundleIdentifier': "com.borjaisern.tmmanager",
        'CFBundleVersion': "1.0.0",
        'CFBundleShortVersionString': "1.0",
        'NSHumanReadableCopyright': 'Copyright © 2025 Borja Isern. All rights reserved.'
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
