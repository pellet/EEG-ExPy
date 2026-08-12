from typing import TYPE_CHECKING

from eegnb.utils.missing import missing_class

MissingExperiment = missing_class(
    "PsychoPy",
    "Stimulus presentation experiments",
    "stimpres",
)

if TYPE_CHECKING:
    from .visual_n170.n170 import VisualN170
    from .visual_p300.p300 import VisualP300
    from .visual_ssvep.ssvep import VisualSSVEP
else:
    try:
        from .visual_n170.n170 import VisualN170
        from .visual_p300.p300 import VisualP300
        from .visual_ssvep.ssvep import VisualSSVEP
    except ImportError:
        VisualN170 = MissingExperiment
        VisualP300 = MissingExperiment
        VisualSSVEP = MissingExperiment

try:
    from psychopy import sound, plugins, prefs
    import os
    import platform
    import logging

    # PTB does not yet support macOS Apple Silicon freely, need to fall back to sounddevice.
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        plugins.scanPlugins()
        success = plugins.loadPlugin('psychopy-sounddevice')

        # Force reload sound module
        import importlib
        import importlib.metadata  # submodule; not implied by `import importlib`
        importlib.reload(sound)

        # loadPlugin() returning True is NOT enough on PsychoPy 2026.x. The audio
        # backend is chosen via `Sound.backend` (a class attribute defaulting to
        # 'ptb') resolved against `Sound.getBackends()`, which scans the
        # *'psychopy.sound.backends'* entry-point group. psychopy-sounddevice
        # declares its entry point under the older *'psychopy.sound'* group, so
        # the backend never reaches the registry: Sound.backend stays 'ptb' and
        # getBackends() returns ['ptb', 'pygame', 'pysound'] while the plugin
        # reports itself loaded. This branch therefore silently did nothing.
        #
        # Bridge the legacy group into the registry, then select it. Wrapped so a
        # failure degrades to PTB rather than breaking `import eegnb.experiments`.
        try:
            from psychopy.sound.sound import Sound as _PsychoPySound
            import psychopy_sounddevice.backend_sounddevice as _sd_backend

            # psychopy does `backends[cls.backend].load().Sound(...)`, but the
            # plugin module only exposes SoundDeviceSound.
            if not hasattr(_sd_backend, 'Sound'):
                _sd_backend.Sound = _sd_backend.SoundDeviceSound

            _orig_get_backends = _PsychoPySound.getBackends.__func__

            def _get_backends_with_legacy_group(cls):
                backends = _orig_get_backends(cls)
                if 'sounddevice' not in backends:
                    for ep in importlib.metadata.entry_points(group='psychopy.sound'):
                        if 'sounddevice' in ep.value:
                            backends['sounddevice'] = importlib.metadata.EntryPoint(
                                name='sounddevice',
                                value=ep.value,
                                group='psychopy.sound.backends',
                            )
                return backends

            _PsychoPySound.getBackends = classmethod(_get_backends_with_legacy_group)
            available = _PsychoPySound.getBackends()
            requested = os.environ.get('EEGNB_AUDIO_BACKEND', '').strip().lower()
            if requested and requested in available:
                _PsychoPySound.backend = requested
            elif 'sounddevice' in available:
                _PsychoPySound.backend = 'sounddevice'
            else:
                logging.warning(
                    "psychopy-sounddevice loaded but its backend entry point was not "
                    "found under either the 'psychopy.sound.backends' or legacy "
                    "'psychopy.sound' group; staying on %s", _PsychoPySound.backend)
        except Exception as e:
            logging.warning(
                "Could not select the sounddevice audio backend (%s: %s); "
                "falling back to PTB.", type(e).__name__, e)

        # Device selection. `sound.setDevice` does not exist in PsychoPy 2026.x,
        # so the previous branch only ever logged "sound.setDevice not available"
        # and left the device unselected. The sounddevice backend takes its
        # output device from sounddevice's own default rather than from prefs,
        # so set it there, matching on name substring.
        audio_device = prefs.hardware.get('audioDevice', 'default')
        if isinstance(audio_device, (list, tuple)):
            audio_device = audio_device[0] if audio_device else 'default'
        if audio_device and audio_device != 'default':
            try:
                import sounddevice as _sd
                matches = [i for i, d in enumerate(_sd.query_devices())
                          if d['max_output_channels'] > 0
                          and audio_device.lower() in d['name'].lower()]
                if matches:
                    _sd.default.device = (_sd.default.device[0], matches[0])
                else:
                    logging.warning(
                        "Requested audio device '%s' matched no output device; "
                        "leaving the system default in place.", audio_device)
            except Exception as e:
                logging.warning("Failed to set audio device to '%s': %s: %s",
                               audio_device, type(e).__name__, e)
    else:
        #change the pref library to PTB and set the latency mode to high precision
        prefs.hardware['audioLib'] = 'PTB'
        prefs.hardware['audioLatencyMode'] = 3
except ImportError:
    import logging
    # logging.warning("PsychoPy not found. Stimulus presentation experiments will not be available.")
    pass

if TYPE_CHECKING:
    from .auditory_oddball.aob import AuditoryOddball
else:
    try:
        from .auditory_oddball.aob import AuditoryOddball
    except ImportError:
        AuditoryOddball = MissingExperiment
