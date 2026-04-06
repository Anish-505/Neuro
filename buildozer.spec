[app]

# (str) Title of your application
title = NeuroMentor

# (str) Package name
package.name = neuromentor

# (str) Package domain (needed for android/ios packaging)
package.domain = org.neuromentor

# (str) Source code where the main.py lives
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,json,pkl

# (str) Application versioning
version = 0.1

# ⚠️ CUSTOM ICON SETTING
icon.filename = %(source.dir)s/logo.png

# ⚠️ LIBRARIES (Do not change)
# Added numpy, scikit-learn, joblib (for the ML backend) and pyjnius (for portrait lock orientation)
requirements = python3,kivy==2.3.0,numpy,scikit-learn,joblib,pyjnius

# (str) Supported orientations (landscape, portrait or all)
orientation = portrait

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK will support.
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 25b

# (bool) Accept SDK license agreements automatically.
android.accept_sdk_license = True

# (str) Android architectures to build for
android.archs = arm64-v8a

# (list) Permissions (added Bluetooth/Location for EEG device streaming instead of Camera)
android.permissions = INTERNET,ACCESS_NETWORK_STATE,ACCESS_WIFI_STATE,BLUETOOTH,BLUETOOTH_ADMIN,ACCESS_FINE_LOCATION,ACCESS_COARSE_LOCATION

# OSX Specifics
osx.python_version = 3
osx.kivy_version = 2.3.0

# (int) fullscreen
fullscreen = 1

[buildozer]
# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1
