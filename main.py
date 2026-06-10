from fireredvad import FireRedVad, FireRedVadConfig

vad_config = FireRedVadConfig(
    use_gpu=False,
    smooth_window_size=5,
    speech_threshold=0.4,
    min_speech_frame=20,
    max_speech_frame=2000,
    min_silence_frame=20,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000)
vad = FireRedVad.from_pretrained("pretrained_models/VAD", vad_config)

result, probs = vad.detect("assets/hello_zh.wav")

print(result)