"""Model-free speech smoke: sine-wave TTS, fake ASR, fake aligner, WAV round trip."""


def test_speech_selftest_passes_without_models():
    from workers import selftest_speech

    assert selftest_speech.run() == 0
