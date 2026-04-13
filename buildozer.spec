[app]
title = Nexus Core
package.name = nexuscore
package.domain = org.test

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json

version = 0.1

# Icon
icon.filename = %(source.dir)s/logo.png

# ✅ ONLY Android-supported libs
requirements = python3,kivy==2.3.0,numpy,requests,pyserial

# ❌ REMOVED:
# opencv → breaks build unless custom recipe
# sklearn → not supported
# joblib → depends on sklearn
# whisper → not supported

orientation = landscape
fullscreen = 1

# Android config
android.api = 33
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True

android.permissions = INTERNET,ACCESS_NETWORK_STATE,ACCESS_WIFI_STATE,CAMERA,RECORD_AUDIO

# Needed for sockets (ESP32 communication)
android.allow_backup = True

[buildozer]
log_level = 2
warn_on_root = 1
