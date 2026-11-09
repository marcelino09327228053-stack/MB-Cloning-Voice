from pathlib import Path
from datetime import datetime
import gc
import os
import threading
import json
import uuid
import time
import html
from urllib.parse import quote
from functools import partial
import numpy as np
import torch
import soundfile as sf
import gradio as gr

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'Qwen3-TTS-12Hz-1.7B-Base'
OUTPUT_DIR = ROOT / 'generated'
OUTPUT_DIR.mkdir(exist_ok=True)
VOICE_DIR = ROOT / 'voices'
VOICE_DIR.mkdir(exist_ok=True)
voice_lock = threading.RLock()
model = None
prompt_cache = {}
model_lock = threading.Lock()
stop_requested = threading.Event()
LANGUAGES = ['Auto', 'Filipino (Auto / experimental)', 'English', 'Chinese', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian']


def saved_voices():
    voices = {}
    for path in sorted(VOICE_DIR.glob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if (VOICE_DIR / (path.stem + '.wav')).is_file():
                voices[path.stem] = data
        except (OSError, ValueError):
            continue
    return voices


def voice_choices():
    voices = saved_voices()
    return [('Upload a voice', 'new')] + [(data['name'], key) for key, data in sorted(voices.items(), key=lambda item: item[1]['name'].casefold())]


def select_voice(key):
    data = saved_voices().get(key)
    if data is None:
        return None, '', False, ''
    return str(VOICE_DIR / (key + '.wav')), data.get('transcript', ''), False, data['name']


def upload_voice(path):
    if not path:
        return None, ''
    # Browser uploads retain the basename, but never expose the source folder.
    filename = Path(str(path)).name
    name = Path(filename).stem.strip() if filename else ''
    if not name.strip('. ') or filename.lower() in {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm'}:
        name = ''
    return str(path), name[:60]


def import_uploaded_voice(path):
    audio, name = upload_voice(path)
    # Browser upload provides a copy, not the original disk path.
    return audio, name, '', ''


def source_target(source_file, name):
    if not source_file:
        return None, None, ''
    source = Path(source_file)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        return None, None, 'Original file unavailable. Choose the original file again.'
    if source.suffix.lower() not in {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm', '.aac'}:
        return None, None, 'Choose an audio file.'
    if not name or any(c in name for c in '<>:"/\\|?*') or any(ord(c) < 32 for c in name) or name.endswith(('.', ' ')) or Path(name).is_reserved():
        return None, None, 'Use a valid filename without slashes or special characters.'
    target = source.with_name(name + source.suffix)
    if target.exists() and not source.samefile(target):
        return None, None, 'A file with that name already exists in the original folder.'
    return source, target, ''


def link_original(key, path):
    source = Path((path or '').strip().strip('"'))
    if not source.is_absolute() or not source.is_file():
        return gr.skip(), 'Paste the complete original audio file path.'
    if source.suffix.lower() not in {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm', '.aac'}:
        return gr.skip(), 'Choose an audio file.'
    with voice_lock:
        data = saved_voices().get(key)
        if not data:
            return gr.skip(), ''
        data['source_file'] = str(source.resolve())
        metadata = VOICE_DIR / (key + '.json')
        temporary = metadata.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        temporary.replace(metadata)
    return time.time_ns(), ''


def save_requirements(name, audio, consent, source_file=""):
    name = (name or '').strip()
    if not audio or not Path(audio).is_file():
        return 'Add a recording.'
    if not name or len(name) > 60:
        return 'Enter a name (up to 60 characters).'
    if any(data['name'].casefold() == name.casefold() for data in saved_voices().values()):
        return 'That name is already saved. Choose another name.'
    if not consent:
        return 'Check permission to save this voice.'
    return source_target(source_file, name)[2]


def save_readiness(name, audio, consent, source_file=""):
    hint = save_requirements(name, audio, consent, source_file)
    return gr.Button(interactive=not hint), hint


def generation_readiness(text, key):
    hint = 'Choose a saved voice with Use.' if key not in saved_voices() else ''
    if not hint and not (text or '').strip():
        hint = 'Enter text to generate speech.'
    if not hint and len(text) > 10000:
        hint = 'Use up to 10,000 characters.'
    return gr.Button(interactive=not hint), hint


def rename_readiness(name, key):
    name = (name or '').strip()
    hint = ''
    if not name or len(name) > 60:
        hint = 'Enter a name (up to 60 characters).'
    elif any(k != key and data['name'].casefold() == name.casefold() for k, data in saved_voices().items()):
        hint = 'That name is already used.'
    if not hint:
        data = saved_voices().get(key, {})
        hint = source_target(data.get('source_file', ''), name)[2]
    return gr.Button(interactive=not hint), hint


def save_voice(name, audio, transcript, consent, source_file=""):
    hint = save_requirements(name, audio, consent, source_file)
    if hint:
        return gr.skip(), hint, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    name = name.strip()
    with voice_lock:
        if any(data['name'].casefold() == name.casefold() for data in saved_voices().values()):
            return gr.skip(), 'That name is already saved.', gr.skip(), gr.skip(), gr.skip(), gr.skip()
        try:
            samples, sr = sf.read(audio, dtype='float32')
        except (OSError, RuntimeError):
            return gr.skip(), 'Cannot read this recording. Try WAV, MP3, or FLAC.', gr.skip(), gr.skip(), gr.skip(), gr.skip()
        if len(samples) == 0:
            return gr.skip(), 'The recording is empty.', gr.skip(), gr.skip(), gr.skip(), gr.skip()
        key = uuid.uuid4().hex
        wav_path = VOICE_DIR / (key + '.wav')
        metadata_path = VOICE_DIR / (key + '.json')
        source, target, hint = source_target(source_file, name)
        if hint:
            return gr.skip(), hint, gr.skip(), gr.skip(), gr.skip(), gr.skip()
        moved = False
        try:
            sf.write(wav_path, samples, sr)
            if source is not None and str(source) != str(target):
                source.rename(target)
                moved = True
            data = {'name': name, 'transcript': (transcript or '').strip()}
            if source is not None:
                data['source_file'] = str(target)
            metadata_path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        except Exception:
            if moved:
                target.rename(source)
            wav_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            return gr.skip(), 'Could not save or rename the original file. Close the file in other apps and try again.', gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return time.time_ns(), f'✓ {name} saved', None, '', '', False


def use_voice(key):
    if key not in saved_voices():
        return ''
    return key


def rename_voice(key, name):
    name = (name or '').strip()
    if not name or len(name) > 60:
        return gr.skip(), gr.skip()
    with voice_lock:
        voices = saved_voices()
        if key not in voices:
            return gr.skip(), ''
        if any(k != key and v['name'].casefold() == name.casefold() for k, v in voices.items()):
            return gr.skip(), gr.skip()
        data = dict(voices[key])
        source, target, hint = source_target(data.get('source_file', ''), name)
        if hint:
            return gr.skip(), gr.skip()
        data['name'] = name
        if source is not None:
            data['source_file'] = str(target)
        path = VOICE_DIR / (key + '.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        moved = False
        try:
            if source is not None and str(source) != str(target):
                source.rename(target)
                moved = True
            temporary.replace(path)
        except OSError:
            if moved:
                target.rename(source)
            temporary.unlink(missing_ok=True)
            return gr.skip(), gr.skip()
    return time.time_ns(), ''


def delete_voice(key, active):
    if not model_lock.acquire(blocking=False):
        raise gr.Error('Wait for generation to finish before deleting a voice.')
    try:
        with voice_lock:
            if key not in saved_voices():
                raise gr.Error('Voice no longer available.')
            trash = VOICE_DIR / '.trash'
            trash.mkdir(exist_ok=True)
            # Keep deleted recordings recoverable, outside the visible library.
            for suffix in ('.json', '.wav'):
                (VOICE_DIR / (key + suffix)).replace(trash / (key + suffix))
            prompt_cache.clear()
        return time.time_ns(), '' if active == key else active, ''
    finally:
        model_lock.release()


def generate_selected(text, key, language, speed, pitch, stability, tokens, progress=gr.Progress()):
    data = saved_voices().get(key)
    _, hint = generation_readiness(text, key)
    if hint:
        return gr.skip(), hint, gr.skip(), gr.skip()
    try:
        result = generate(text, str(VOICE_DIR / (key + '.wav')), data.get('transcript', ''),
                          language, True, speed, pitch, stability, tokens, progress)
    except gr.Error as exc:
        return gr.skip(), str(exc.message), gr.skip(), gr.skip()
    if isinstance(result[0], str):
        return gr.Audio(value=result[0], visible=True), result[1], result[2], result[0]
    return (*result, gr.skip())


def preview_html(key):
    source = '/gradio_api/file=' + quote(str((VOICE_DIR / (key + '.wav')).resolve()).replace('\\', '/'), safe='/:')
    return '<audio preload="none" src="' + html.escape(source, quote=True) + '"></audio><div class="preview-actions"><button type="button" data-action="play">Play</button><button type="button" data-action="pause" disabled>Pause</button><span role="status" class="preview-error"></span></div>'


PREVIEW_JS = """
const audio = element.querySelector('audio');
const play = element.querySelector('[data-action="play"]');
const pause = element.querySelector('[data-action="pause"]');
const error = element.querySelector('.preview-error');
play.addEventListener('click', async () => {
    if (window.mbVoicePreview && window.mbVoicePreview !== audio) window.mbVoicePreview.pause();
    window.mbVoicePreview = audio;
    try { await audio.play(); error.textContent = ''; }
    catch { error.textContent = 'Could not play'; }
});
pause.addEventListener('click', () => audio.pause());
audio.addEventListener('play', () => { play.disabled = true; pause.disabled = false; });
audio.addEventListener('pause', () => { play.disabled = false; pause.disabled = true; });
audio.addEventListener('ended', () => { play.disabled = false; pause.disabled = true; });
"""


def history_files():
    return [p.name for p in sorted(OUTPUT_DIR.glob('mb_voice_*.wav'), key=lambda p: p.stat().st_mtime, reverse=True)]


def history_update():
    names = history_files()
    return gr.Radio(choices=names, value=names[0] if names else None)


def load_history(name):
    if not name or name not in history_files():
        return None
    return str(OUTPUT_DIR / name)


def clear_result():
    return gr.Audio(value=None, visible=False), 'Ready · Result cleared; saved audio is in History.', ''


def delete_recording(name, current):
    if not name or name not in history_files():
        return gr.skip(), gr.skip(), 'Choose a recording first.', gr.skip(), gr.skip(), gr.skip()
    path = OUTPUT_DIR / name
    try:
        path.unlink()
    except OSError:
        return gr.skip(), gr.skip(), 'Could not delete this recording. Close the file in other apps and try again.', gr.skip(), gr.skip(), gr.skip()
    is_current = bool(current) and Path(current).name == name
    remaining = history_files()
    next_audio = load_history(remaining[0]) if remaining else None
    return (history_update(), next_audio, 'Recording permanently deleted.',
            gr.Audio(value=None, visible=False) if is_current else gr.skip(),
            'Ready · Recording deleted.' if is_current else gr.skip(),
            '' if is_current else gr.skip())


def gpu_status():
    if not torch.cuda.is_available():
        return '● CUDA unavailable · GPU generation requires CUDA-enabled PyTorch'
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**3
    return f'● {torch.cuda.get_device_name(0)} · VRAM {used:.2f} / {total / 1024**3:.1f} GB · Model {"loaded" if model is not None else "on standby"}'


def open_folder():
    os.startfile(str(OUTPUT_DIR))
    return 'Output folder opened on this computer.'


def unload_model():
    global model
    if not model_lock.acquire(blocking=False):
        return 'Generation is running. Stop or wait before unloading.'
    try:
        model = None
        prompt_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 'Model unloaded. GPU memory released.'
    finally:
        model_lock.release()


def request_stop():
    stop_requested.set()
    return 'Stop requested. Finishing the current model step…'


def audio_info(path):
    if not path:
        return 'WAV / MP3 / FLAC · 5–30 seconds recommended'
    try:
        info = sf.info(path)
        return f'{Path(path).name} · {info.duration:.1f}s · {info.samplerate:,} Hz · {info.channels} channel(s)'
    except Exception:
        return Path(path).name


def generate(text, audio, transcript, language, consent, speed, pitch, stability, tokens, progress=gr.Progress()):
    global model
    if not consent:
        raise gr.Error('Confirm that you own or have permission to clone this voice.')
    if not text or not text.strip():
        raise gr.Error('Enter text to speak.')
    if not audio or not Path(audio).is_file():
        raise gr.Error('Upload or record a reference voice first.')
    if len(text) > 10000:
        raise gr.Error('Maximum 10,000 characters per generation.')
    if not (MODEL_PATH / 'model.safetensors').exists():
        raise gr.Error('Voice model not found. Run Download Qwen Voice Model.cmd.')
    if not torch.cuda.is_available():
        raise gr.Error('CUDA is unavailable. Run Test Qwen GPU.cmd to check the installation.')
    with model_lock:
        started = time.perf_counter()
        stop_requested.clear()
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class StopOnRequest(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return torch.full((input_ids.shape[0],), stop_requested.is_set(), dtype=torch.bool, device=input_ids.device)

            if model is None:
                progress(0.05, desc='Loading Qwen3-TTS onto GPU…')
                from qwen_tts import Qwen3TTSModel
                model = Qwen3TTSModel.from_pretrained(str(MODEL_PATH), device_map='cuda:0', dtype=torch.bfloat16)
            if stop_requested.is_set():
                return gr.skip(), 'Stopped.', history_update()
            language = 'Auto' if language.startswith('Filipino') else language
            supported = model.model.get_supported_languages()
            if language != 'Auto' and language.lower() not in [str(x).lower() for x in supported]:
                raise gr.Error(f'Language {language} is not supported by this model.')
            progress(0.2, desc='Generating speech… First generation can take several minutes.')
            has_transcript = bool(transcript and transcript.strip())
            with torch.inference_mode():
                reference = Path(audio)
                ref_text = transcript.strip() if has_transcript else None
                cache_key = (str(reference.resolve()), reference.stat().st_mtime_ns, reference.stat().st_size, ref_text)
                if cache_key not in prompt_cache:
                    prompt_cache.clear()
                    prompt_cache[cache_key] = model.create_voice_clone_prompt(
                        ref_audio=audio, ref_text=ref_text, x_vector_only_mode=not has_transcript)
                if stop_requested.is_set():
                    return gr.skip(), 'Stopped.', history_update()
                wavs, sr = model.generate_voice_clone(
                    text=text.strip(), language=language, voice_clone_prompt=prompt_cache[cache_key],
                    max_new_tokens=int(tokens),
                    temperature=1.2 - float(stability) * 0.7,
                    stopping_criteria=StoppingCriteriaList([StopOnRequest()]),
                )
            if stop_requested.is_set():
                return gr.skip(), 'Stopped. No output saved.', history_update()
            samples = np.asarray(wavs[0], dtype=np.float32)
            if speed != 1 or pitch != 0:
                progress(0.9, desc='Applying speed and pitch…')
                import librosa
                if pitch != 0:
                    samples = librosa.effects.pitch_shift(samples, sr=sr, n_steps=float(pitch))
                if speed != 1:
                    samples = librosa.effects.time_stretch(samples, rate=float(speed))
            if stop_requested.is_set():
                return gr.skip(), 'Stopped. No output saved.', history_update()
            path = OUTPUT_DIR / ('mb_voice_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.wav')
            sf.write(str(path), samples, sr)
            return str(path), f'✓ {path.name} · Audio {len(samples) / sr:.1f}s · Processing {time.perf_counter() - started:.1f}s', history_update()
        except gr.Error:
            raise
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error('GPU memory insufficient. Close other GPU applications or unload the model in Settings.')
        except Exception as exc:
            raise gr.Error(f'Generation failed: {exc}')


CSS = '''
:root,.dark,.gradio-container {color-scheme:dark;--body-background-fill:transparent!important;--block-background-fill:transparent!important;--input-background-fill:#ffffff06!important;--block-border-color:#ffffff20!important;--input-border-color:#ffffff20!important;--body-text-color:#eeeaf9!important;--block-label-background-fill:transparent!important;--block-label-text-color:#b9b4d0!important;--input-text-color:#f5f2ff!important;}
html,body {margin:0!important;background:#0c1021!important;}
body {background:radial-gradient(ellipse at 10% 15%,#7b3bb566,transparent 50%),radial-gradient(ellipse at 90% 25%,#157c9c55,transparent 48%),radial-gradient(ellipse at 55% 100%,#b3487e44,transparent 50%),#0c1021!important;background-attachment:fixed!important;}
.gradio-container {max-width:none!important;padding:14px!important;background:transparent!important;font-family:'Segoe UI',sans-serif!important;color:#f2effa!important;min-height:100dvh!important;gap:0!important;}
#masthead {height:46px;box-sizing:border-box;display:flex;align-items:center;padding:0 20px;background:#ffffff06;border:1px solid #ffffff20;border-radius:14px 14px 0 0;backdrop-filter:blur(24px);}
.brand {font-size:17px;font-weight:750;letter-spacing:.3px;}.brand span {color:#d0b5ff;}
#workspace {gap:0!important;background:#13152b66;border:1px solid #ffffff20;border-top:0;border-radius:0 0 14px 14px;backdrop-filter:blur(24px);box-shadow:0 20px 70px #00000035;}
#sidebar {flex:0 0 165px!important;min-width:165px!important;padding:18px 12px!important;border-right:1px solid #ffffff14;gap:8px!important;}
#main {padding:18px!important;min-width:0!important;}
.page-heading {font-size:21px;font-weight:650;margin:0 0 10px;letter-spacing:-.4px;}
#studio-layout {gap:18px!important;align-items:stretch;}
#editor-pane {gap:8px!important;min-width:0!important;}
#voice-pane {flex:0 0 350px!important;min-width:350px!important;padding:14px!important;background:#ffffff05!important;border:1px solid #ffffff1c;border-radius:12px;gap:10px!important;}
.panel-heading {font-size:15px;font-weight:650;}
#script {border:1px solid #ffffff22!important;border-radius:12px;background:#ffffff04!important;overflow:hidden;}
#script textarea {background:transparent!important;color:#f5f2ff!important;font-size:17px!important;line-height:1.7!important;padding:18px!important;resize:none!important;}
#script:focus-within {border-color:#c0a1fa88!important;box-shadow:0 0 0 3px #ab83ed0c;}
#counter {text-align:right;color:#aaa2be;font-size:11px;min-height:16px;}
#editor-actions {gap:8px!important;}
.gradio-container button:not([role='tab']):not([aria-label]),.preview-actions button {height:34px!important;min-height:34px!important;padding:0 12px!important;border-radius:8px!important;border:1px solid #d6c7ff30!important;background:linear-gradient(160deg,#b7a3e51c,#e5ddff08)!important;color:#eee9fa!important;font-family:inherit!important;font-size:12px!important;font-weight:550!important;box-shadow:inset 0 1px #ffffff0a!important;transition:background .18s,border-color .18s;}
.gradio-container button:not([role='tab']):hover,.preview-actions button:hover {border-color:#d2b5ff77!important;background:#ad86ed24!important;}
#generate-btn {background:linear-gradient(100deg,#9672d655,#6e88d833)!important;border-color:#b89dec66!important;}
#sidebar .primary {background:#b49cdd22!important;border-color:#cdb1ff44!important;}
#sidebar button {text-align:left;}
.voice-card button.use-active {background:#18784588!important;border-color:#57d79588!important;color:#caffdf!important;}
.gradio-container button:disabled,.preview-actions button:disabled {opacity:.4!important;cursor:default;}
#voice-list {padding:0!important;gap:8px!important;overflow:visible!important;max-height:none!important;}
.voice-card {padding:10px!important;border:1px solid #ffffff1c;border-radius:10px;background:#ffffff04;gap:6px!important;}
.voice-card.voice-active {border-color:#55c78b66;background:#24b57209;}
.voice-name {font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.voice-card .row {gap:6px!important;align-items:center;}
.voice-preview {min-width:0!important;flex:2!important;}
.preview-actions {display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.preview-actions button {width:100%;cursor:pointer;}
.preview-error {font-size:10px;color:#ffb2bd;grid-column:1/-1;}.preview-error:empty {display:none;}
.voice-pagination {align-items:center;gap:6px!important;}.page-count {text-align:center;font-size:12px;color:#bfb4d4;}
#voice-tabs {border:0!important;background:transparent!important;}
#voice-tabs .tabitem {padding:10px 0 0!important;border:0!important;background:transparent!important;}
#voice-tabs [role='tablist'] {background:transparent!important;border-color:#ffffff16;}
#voice-tabs [role='tab'] {font-size:12px!important;padding:8px 12px!important;color:#bdb2d2;}
#voice-tabs [aria-selected='true'] {color:#efdcff!important;border-bottom-color:#c49af3!important;}
#reference-audio {min-height:100px!important;background:#ffffff04!important;border:1px dashed #c6b5e833!important;border-radius:10px;}
#reference-audio .wrap {min-height:100px!important;}
#settings-row {gap:8px!important;background:transparent!important;}
.gradio-container .block,.gradio-container .form {background:transparent!important;}
.gradio-container input,.gradio-container textarea {background:#ffffff06!important;color:#f2edff!important;border-radius:8px!important;}
.gradio-container label {background:transparent!important;}
#result-panel {background:#ffffff04!important;border:1px solid #ffffff18;border-radius:10px;padding:10px!important;gap:6px!important;}
#status textarea {font-size:12px!important;color:#a5e3bd!important;padding:7px 10px!important;min-height:30px!important;background:transparent!important;}
#footer {font-size:11px;color:#b2a9c9;padding:8px 12px!important;}
.section-title {font-size:20px;font-weight:600;margin-bottom:12px;}.section-title b {display:none;}
footer {display:none!important;}
input:focus-visible,textarea:focus-visible,button:focus-visible {outline:2px solid #c6a4f5!important;outline-offset:2px;}
@media(min-width:1100px) and (min-height:750px) {
html,body {height:100%;overflow:hidden!important;}
.gradio-container {height:100dvh!important;min-height:0!important;}
#workspace {height:calc(100dvh - 136px)!important;min-height:0!important;}
#main,#studio-layout,#editor-pane {min-height:0!important;}
#studio-layout {height:calc(100dvh - 190px)!important;}
#script {flex:1!important;min-height:100px!important;}
#script>div,#script textarea {height:100%!important;min-height:0!important;}
#script textarea {overflow-y:auto!important;}
}
@media(max-width:1099px),(max-height:749px) {#script textarea {min-height:200px!important;} #sidebar {min-height:0!important;} }
@media(max-width:850px) {#workspace,#studio-layout {flex-direction:column!important;} #sidebar {flex:auto!important;border-right:0;min-width:0!important;} #voice-pane {flex:auto!important;min-width:0!important;} }
@media(prefers-reduced-motion:reduce) {* {transition:none!important;animation:none!important;}}

/* Fit the Gradio wrappers as well as the visible panels. */
body .gradio-container.gradio-container {min-height:0!important;}
.gradio-container > .main {padding:0!important;max-width:none!important;}
.gradio-container > .main > .wrap {max-width:none!important;}
.gradio-container main.contain > .column {gap:0!important;}
#main > .column {gap:0!important;}
#counter {padding:0 4px!important;height:20px!important;}
#result-panel {flex:0 0 auto!important;min-height:0!important;}
#status {padding:0!important;}
.voice-card .block {padding:0!important;min-height:0!important;}
#voice-list {min-height:0!important;}
#voice-pane > .block {padding:0!important;}
#voice-pane .tabitem .form {gap:8px!important;}
#voice-pane label {font-size:12px!important;color:#c7bfd8!important;}
@media(min-width:1100px) and (min-height:750px) {
html,body,gradio-app {height:100dvh!important;min-height:0!important;overflow:hidden!important;}
body .gradio-container.gradio-container {height:100dvh!important;min-height:0!important;box-sizing:border-box!important;}
#workspace {height:calc(100dvh - 136px)!important;}
#main {height:100%!important;box-sizing:border-box;}
#studio-layout {height:calc(100dvh - 216px)!important;}
#editor-pane,#voice-pane {height:100%!important;min-height:0!important;box-sizing:border-box!important;}
#editor-pane > .form {flex:1 1 0%!important;min-height:0!important;}
#script,#script>div,#script textarea {height:100%!important;min-height:0!important;box-sizing:border-box!important;}
#voice-pane {align-self:stretch;}
}

#header-block {padding:0!important;min-height:0!important;}
#editor-pane > .form {border:0!important;border-radius:12px;overflow:hidden;}
#script {position:relative;}
@media(min-width:1100px) and (min-height:750px) {
#script textarea {position:absolute!important;inset:0;width:100%!important;}
}
.voice-preview .html-container {padding:0!important;}

#voice-pane {flex-wrap:nowrap!important;}
#voice-pane > .block {flex:0 0 auto!important;}
#voice-tabs {flex:1 1 auto!important;min-height:0!important;width:100%!important;}
#voice-tabs .tabitem > .column {gap:6px!important;flex-wrap:nowrap!important;}
#voice-tabs .tabitem .block {padding:4px!important;}
#voice-tabs .tabitem .form {gap:6px!important;}
@media(min-width:1100px) and (min-height:750px) {
#script textarea {position:static!important;inset:auto!important;height:calc(100dvh - 380px)!important;max-height:calc(100dvh - 380px)!important;}
#editor-pane:has(#result-audio) #script textarea {height:calc(100dvh - 500px)!important;max-height:calc(100dvh - 500px)!important;}
}

#result-toolbar {gap:8px!important;align-items:center!important;}
#result-toolbar > .form {flex:1!important;min-width:0!important;}
#clear-result {align-self:center;}
#rename-input {padding:6px!important;}

/* Neon glass controls: the same silhouette for every action. */
#workspace {--neon-rgb:168,130,238;}
#workspace button:not([role='tab']):not([aria-label]) {
height:34px!important;min-height:34px!important;box-sizing:border-box;flex-wrap:nowrap!important;white-space:nowrap;align-items:center;justify-content:center;gap:7px;
border-radius:10px!important;border:1px solid rgba(var(--neon-rgb),.58)!important;
background:linear-gradient(160deg,rgba(var(--neon-rgb),.18),rgba(var(--neon-rgb),.055))!important;
color:#f5efff!important;box-shadow:inset 0 1px 0 #ffffff12,0 0 8px rgba(var(--neon-rgb),.08)!important;
backdrop-filter:blur(10px);font-weight:600!important;cursor:pointer;
transition:border-color .16s,box-shadow .16s,background .16s,transform .1s!important;
}
#workspace button:not([role='tab']):not([aria-label]):hover:not(:disabled) {
border-color:rgba(var(--neon-rgb),.95)!important;
background:linear-gradient(160deg,rgba(var(--neon-rgb),.32),rgba(var(--neon-rgb),.14))!important;
box-shadow:inset 0 1px 0 #ffffff25,0 0 5px rgba(var(--neon-rgb),.5),0 0 17px rgba(var(--neon-rgb),.25)!important;
}
#workspace button:not([role='tab']):not([aria-label]):active:not(:disabled) {
transform:translateY(1px);background:linear-gradient(160deg,rgba(var(--neon-rgb),.55),rgba(var(--neon-rgb),.3))!important;
box-shadow:inset 0 2px 6px #00000040,0 0 10px rgba(var(--neon-rgb),.3)!important;
}
#workspace button:not([role='tab']):not([aria-label]):disabled {
--neon-rgb:118,113,140;opacity:.48!important;cursor:not-allowed;transform:none;
background:#9790b20a!important;border-color:#9a91b52b!important;color:#c5bed1!important;box-shadow:none!important;
}
#workspace button.use-voice {--neon-rgb:169,78,255;}
#workspace button.use-active {--neon-rgb:58,219,132;background:linear-gradient(160deg,#219e5b66,#18734433)!important;border-color:#4bdd9499!important;color:#d8ffe9!important;}
#workspace button[data-action='play'],#workspace button[data-action='pause'] {--neon-rgb:67,155,255;}
#workspace button.neon-rename {--neon-rgb:179,146,232;}
#workspace button.neon-delete {--neon-rgb:255,71,145;}
#workspace button.neon-save {--neon-rgb:137,113,242;}
#workspace #generate-btn {--neon-rgb:166,88,255;}
#workspace #stop-btn {--neon-rgb:154,143,190;}
#workspace #sidebar button {justify-content:flex-start;}
#workspace button::before {filter:drop-shadow(0 0 3px rgba(var(--neon-rgb),.5));}
#workspace button:disabled::before {filter:none;}
@media(prefers-reduced-motion:reduce) {#workspace button {transition:none!important;transform:none!important;}}
#workspace button.use-voice::before,#workspace button.use-active::before{content:"";width:16px;height:16px;flex:0 0 16px;background:rgb(var(--neon-rgb));mask:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22m12%203%202.4%206.6L21%2012l-6.6%202.4L12%2021l-2.4-6.6L3%2012l6.6-2.4L12%203Z%22%2F%3E%3Cpath%20d%3D%22m20%202%20.8%202.2L23%205l-2.2.8L20%208l-.8-2.2L17%205l2.2-.8L20%202Z%22%2F%3E%3C%2Fsvg%3E") center/contain no-repeat;}
#workspace button[data-action="play"]::before{content:"";width:16px;height:16px;flex:0 0 16px;background:rgb(var(--neon-rgb));mask:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22m7%203%2014%209L7%2021V3Z%22%2F%3E%3C%2Fsvg%3E") center/contain no-repeat;}
#workspace button[data-action="pause"]::before{content:"";width:16px;height:16px;flex:0 0 16px;background:rgb(var(--neon-rgb));mask:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M8%204v16M16%204v16%22%2F%3E%3C%2Fsvg%3E") center/contain no-repeat;}
#workspace button.neon-rename::before{content:"";width:16px;height:16px;flex:0 0 16px;background:rgb(var(--neon-rgb));mask:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22m15%205%204%204M4%2020l4-1L21%206a2.8%202.8%200%200%200-4-4L4%2015l-1%206%205-2%22%2F%3E%3C%2Fsvg%3E") center/contain no-repeat;}
#workspace button.neon-delete::before{content:"";width:16px;height:16px;flex:0 0 16px;background:rgb(var(--neon-rgb));mask:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M3%206h18M9%206V3h6v3M5%206l1%2015h12l1-15M10%2010v7M14%2010v7%22%2F%3E%3C%2Fsvg%3E") center/contain no-repeat;}
#workspace button.neon-save::before{content:"";width:16px;height:16px;flex:0 0 16px;background:rgb(var(--neon-rgb));mask:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M4%203h13l4%204v14H3V3h1ZM7%203v6h9V3M7%2021v-8h10v8%22%2F%3E%3C%2Fsvg%3E") center/contain no-repeat;}
#workspace button#generate-btn::before{content:"";width:16px;height:16px;flex:0 0 16px;background:rgb(var(--neon-rgb));mask:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22m13%202-9%2012h7l-1%208%2010-13h-8l1-7Z%22%2F%3E%3C%2Fsvg%3E") center/contain no-repeat;}
#workspace button#stop-btn::before{content:"";width:16px;height:16px;flex:0 0 16px;background:rgb(var(--neon-rgb));mask:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Crect%20x%3D%225%22%20y%3D%225%22%20width%3D%2214%22%20height%3D%2214%22%20rx%3D%222%22%2F%3E%3C%2Fsvg%3E") center/contain no-repeat;}
#workspace button#clear-result::before{content:"";width:16px;height:16px;flex:0 0 16px;background:rgb(var(--neon-rgb));mask:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22m14%203%207%207-10%2011H7l-5-5L14%203ZM7%2010l8%208M11%2021h11%22%2F%3E%3C%2Fsvg%3E") center/contain no-repeat;}

#generation-hint,#save-hint {padding:0 4px!important;font-size:11px!important;color:#bcb2d1;min-height:0!important;}
#generation-hint:has(.prose:empty),#save-hint:has(.prose:empty) {display:none;}

#voice-permission {border:1px solid transparent!important;border-radius:8px!important;padding:6px!important;transition:background .15s,border-color .15s;}
#voice-permission:has(input:checked) {background:#22c55e19!important;border-color:#4ade8077!important;}
#voice-permission:has(input:checked) label,#voice-permission:has(input:checked) span {color:#bbf7d0!important;}
#voice-permission input[type='checkbox'] {appearance:none!important;-webkit-appearance:none!important;width:19px!important;height:19px!important;min-width:19px!important;border:1px solid #b3a9cb77!important;border-radius:5px!important;background:#ffffff09!important;display:inline-grid!important;place-content:center;cursor:pointer;}
#voice-permission input[type='checkbox']:checked {background:#22c55e!important;border-color:#86efac!important;box-shadow:0 0 10px #22c55e55!important;}
#voice-permission input[type='checkbox']:checked::after {content:'';display:block;width:9px;height:5px;border-left:2px solid #052e16;border-bottom:2px solid #052e16;transform:translateY(-1px) rotate(-45deg);}
#voice-permission input:focus-visible {outline:2px solid #86efac!important;outline-offset:3px;}

/* Five voice cards on a full desktop window; smaller windows keep the list contained. */
#voice-list {max-height:min(632px, calc(100dvh - 320px))!important;overflow-y:auto!important;overflow-x:hidden!important;overscroll-behavior:contain;scrollbar-width:thin;scrollbar-color:#9e70d9 transparent;padding-right:5px!important;}
#voice-list > .column {gap:8px!important;flex-wrap:nowrap!important;}
#voice-list .voice-card {height:120px!important;min-height:120px!important;flex:0 0 120px!important;box-sizing:border-box;padding:8px!important;gap:5px!important;}
#voice-list .voice-card .block,#voice-list .voice-card .html-container {padding:0!important;min-height:0!important;}
#voice-list .voice-name {font-size:18px!important;line-height:22px!important;font-weight:700!important;color:#91f2b5!important;text-shadow:0 0 10px #45c98218;}
#voice-list .voice-card .row {flex:0 0 auto!important;min-height:34px!important;}
#voice-list:has(#rename-input) {max-height:100%!important;}

#voice-pane > .block .html-container {padding:0!important;}
#voice-tabs > .tab-wrapper {margin:0!important;}

/* Quiet voice names; actions appear only when a voice is opened. */
#voice-list {max-height:min(322px, calc(100dvh - 320px))!important;}
#workspace #voice-list .voice-dropdown {padding:0!important;border:1px solid #ffffff19!important;border-radius:10px!important;background:#ffffff03!important;flex:0 0 auto!important;overflow:hidden;}
#workspace #voice-list .voice-dropdown > button {height:56px!important;min-height:56px!important;width:100%;padding:0 12px!important;display:flex!important;flex-wrap:nowrap!important;justify-content:space-between!important;gap:10px!important;border:0!important;border-radius:0!important;background:transparent!important;box-shadow:none!important;transform:none!important;}
#workspace #voice-list .voice-dropdown > button:hover {background:#ffffff06!important;border:0!important;box-shadow:none!important;}
#workspace #voice-list .voice-dropdown > button span {font-size:17px!important;font-weight:650!important;color:#a8e7bf!important;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
#workspace #voice-list .voice-dropdown > button svg {width:14px;height:14px;color:#b6a9ca;flex-shrink:0;}
#workspace #voice-list .voice-dropdown.voice-active {border-color:#49ba7b66!important;background:#36b26c08!important;}
#workspace #voice-list .voice-dropdown.voice-active > button span {color:#86efac!important;}
#workspace #voice-list .voice-dropdown .row {gap:6px!important;align-items:center;}
#workspace #voice-list .voice-dropdown .block,#workspace #voice-list .voice-dropdown .html-container {padding:0!important;}
#workspace #voice-list .voice-dropdown > .wrap,#workspace #voice-list .voice-dropdown > .content {padding:8px!important;}

/* Let Saved use all remaining panel height before it starts scrolling. */
#voice-list {max-height:calc(100dvh - 320px)!important;}
@media(min-width:1100px) and (min-height:750px) {
#voice-tabs {display:flex!important;flex-direction:column!important;overflow:hidden!important;}
#voice-tabs > .tab-wrapper {flex:0 0 auto!important;}
#saved-voices-tab {flex:1 1 0%!important;min-height:0!important;overflow:hidden!important;}
#saved-voices-tab > .column {flex:1 1 0%!important;min-height:0!important;}
#voice-list {flex:1 1 0%!important;max-height:none!important;min-height:0!important;}
}

#history-layout {gap:20px!important;align-items:flex-start;}
#history-list {max-height:calc(100dvh - 275px);overflow-y:auto;scrollbar-width:thin;scrollbar-color:#9e70d9 transparent;}
#history-list .wrap {display:flex!important;flex-direction:column!important;gap:8px!important;}
#history-list label {width:100%;box-sizing:border-box;padding:12px!important;border:1px solid #ffffff1c!important;border-radius:9px;background:#ffffff03!important;overflow-wrap:anywhere;}
#history-list label:has(input:checked) {border-color:#b79ada80!important;background:#b894e712!important;}
#history-list label span {font-size:13px!important;}

#result-playback {gap:14px!important;align-items:center!important;}
#workspace #download-audio,#workspace #download-audio button {--neon-rgb:64,218,137;height:48px!important;min-height:48px!important;padding:0 18px!important;border:1px solid #7df0b1aa!important;background:linear-gradient(145deg,#178b50,#12683f)!important;color:#eefff5!important;font-size:15px!important;font-weight:700!important;border-radius:10px!important;box-shadow:0 0 14px #2ed98022!important;white-space:nowrap;align-self:center;}
#workspace #download-audio:hover,#workspace #download-audio button:hover {background:linear-gradient(145deg,#20a562,#178b50)!important;box-shadow:0 0 19px #38db8055!important;}
#workspace #download-audio::before {content:'↓';font-size:23px;font-weight:700;line-height:1;margin-right:4px;}

#text-tools {gap:10px!important;align-items:center!important;flex:0 0 auto!important;}
#text-tools #counter {align-self:center;}
#workspace #clear-text {height:30px!important;min-height:30px!important;font-size:12px!important;}

#text-tools > .block {padding:0!important;}
.editor-label {font-size:13px;color:#b7afca;}
#workspace #clear-text {--neon-rgb:64,218,137;background:linear-gradient(145deg,#178b50,#12683f)!important;border-color:#7df0b199!important;color:#eefff5!important;}
#workspace #clear-text:hover:not(:disabled) {background:linear-gradient(145deg,#20a562,#178b50)!important;box-shadow:0 0 12px #38db8033!important;}
#workspace #clear-text:disabled {background:#1a513330!important;border-color:#4e9b6a44!important;color:#92bca3!important;}

'''


def section(number, title):
    gr.HTML(f'<div class="section-title"><b>{number}</b>{title}</div>')


with gr.Blocks(title='MB Cloning Voice', analytics_enabled=False) as demo:
    gr.HTML('<header id="masthead"><div class="brand"><span>MB</span> CLONING VOICE</div></header>', elem_id='header-block')
    with gr.Row(elem_id='workspace'):
        with gr.Column(scale=1, min_width=170, elem_id='sidebar'):
            nav_clone = gr.Button('Text to Speech', variant='primary')
            nav_history = gr.Button('History')
            nav_settings = gr.Button('Settings')
        with gr.Column(scale=5, min_width=320, elem_id='main'):
            with gr.Column(visible=True) as clone_page:
                gr.HTML('<div class="page-heading">Text to Speech</div>')
                with gr.Row(elem_id='studio-layout'):
                    with gr.Column(scale=7, min_width=330, elem_id='editor-pane'):
                        with gr.Row(elem_id='text-tools'):
                            gr.HTML('<div class="editor-label">Text</div>', scale=5)
                            clear_text = gr.Button('Clear text', size='sm', scale=0, min_width=95, interactive=False, elem_id='clear-text')
                        script = gr.Textbox(label='Text', show_label=False, lines=8, max_lines=8, max_length=10000, placeholder='Write something to say…', elem_id='script')
                        counter = gr.HTML('0 / 10,000', elem_id='counter')
                        with gr.Row(elem_id='editor-actions'):
                            button = gr.Button('Generate', variant='primary', scale=4, elem_id='generate-btn', interactive=False)
                            stop = gr.Button('Stop', scale=1, elem_id='stop-btn')
                        generation_hint = gr.Markdown('Choose a saved voice with Use.', elem_id='generation-hint')
                        with gr.Column(elem_id='result-panel'):
                            current_output = gr.State('')
                            with gr.Row(elem_id='result-toolbar'):
                                status = gr.Textbox(value='Ready', show_label=False, interactive=False, elem_id='status', scale=5)
                                clear_audio = gr.Button('Clear', size='sm', scale=0, min_width=85, elem_id='clear-result', interactive=False)
                            with gr.Row(elem_id='result-playback'):
                                output = gr.Audio(show_label=False, type='filepath', interactive=False, buttons=[], visible=False, elem_id='result-audio', scale=5, min_width=240)
                                download_audio = gr.DownloadButton('Download audio', value=None, visible=False, interactive=False, scale=0, min_width=170, elem_id='download-audio')
                    with gr.Column(scale=3, min_width=280, elem_id='voice-pane'):
                        gr.HTML('<div class="panel-heading">Voices</div>')
                        active_voice = gr.State('')
                        library_version = gr.State(0)
                        editing_voice = gr.State('')
                        with gr.Tabs(elem_id='voice-tabs'):
                            with gr.Tab('Saved', elem_id='saved-voices-tab'):
                                with gr.Column(elem_id='voice-list'):
                                    @gr.render(inputs=[library_version, active_voice, editing_voice])
                                    def render_voices(version, active, editing):
                                        voices = saved_voices()
                                        if editing in voices:
                                            gr.Markdown('**Rename voice**')
                                            source = voices[editing].get('source_file', '')
                                            gr.Markdown('Original: ' + Path(source).name if source else 'Original file not linked. Link its path to rename the original too.')
                                            with gr.Accordion('Link original file (optional)', open=False):
                                                original_path = gr.Textbox(value=source, label='Original file path', placeholder=r'D:\Recordings\voice.wav')
                                                locate_original = gr.Button('Link file', size='sm')
                                            new_name = gr.Textbox(value=voices[editing]['name'], label='Name', max_length=60, elem_id='rename-input')
                                            with gr.Row():
                                                rename_save = gr.Button('Save', elem_id='rename-save', elem_classes='neon-save')
                                                rename_cancel = gr.Button('Cancel')
                                            rename_hint = gr.Markdown('')
                                            locate_original.click(partial(link_original, editing), inputs=original_path, outputs=[library_version, rename_hint], queue=True, api_visibility='private')
                                            new_name.change(partial(rename_readiness, key=editing), inputs=new_name, outputs=[rename_save, rename_hint], queue=True, show_progress='hidden')
                                            rename_save.click(partial(rename_voice, editing), inputs=new_name, outputs=[library_version, editing_voice], queue=True, api_visibility='private')
                                            rename_cancel.click(lambda: '', outputs=editing_voice, queue=True)
                                            return
                                        if not voices:
                                            gr.Markdown('No voices yet. Add one below.')
                                        ordered = sorted(voices.items(), key=lambda item: item[1]['name'].casefold())
                                        for key, data in ordered:
                                            with gr.Accordion(data['name'], open=False, key='voice-menu-' + key, preserved_by_key=['open'], elem_classes=['voice-dropdown', 'voice-active' if key == active else 'voice-idle']):
                                                with gr.Row():
                                                    use = gr.Button('In use' if key == active else 'Use', size='sm', min_width=65, elem_classes=['use-active' if key == active else 'use-voice'])
                                                    gr.HTML(preview_html(key), js_on_load=PREVIEW_JS, elem_classes='voice-preview')
                                                with gr.Row():
                                                    rename = gr.Button('Rename', size='sm', min_width=65, elem_classes='neon-rename')
                                                    delete = gr.Button('Delete', size='sm', min_width=65, elem_classes='neon-delete')
                                                use.click(lambda k=key: use_voice(k), outputs=active_voice, queue=True)
                                                rename.click(lambda k=key: k, outputs=editing_voice, queue=True)
                                                delete.click(lambda selected, k=key: delete_voice(k, selected), inputs=active_voice, outputs=[library_version, active_voice, editing_voice], api_visibility='private')
                            with gr.Tab('Add voice'):
                                source_file = gr.State('')
                                add_voice_upload = gr.UploadButton('Add voice', file_types=['audio'], type='filepath', size='sm', elem_id='add-voice-upload')
                                audio = gr.Audio(label='Recording', type='filepath', sources=['microphone'], buttons=[], editable=False, elem_id='reference-audio')
                                voice_name = gr.Textbox(label='Name', max_length=60, elem_id='voice-name-input')
                                with gr.Accordion('Transcript (optional)', open=False):
                                    transcript = gr.Textbox(show_label=False, lines=1)
                                consent = gr.Checkbox(label='I have permission to use this voice.', elem_id='voice-permission')
                                save = gr.Button('Save voice', elem_classes='neon-save', interactive=False, elem_id='save-voice')
                                voice_status = gr.Markdown('Add a recording.', elem_id='save-hint')
            with gr.Column(visible=False) as history_page:
                section('◷', 'History')
                with gr.Row(elem_id='history-layout'):
                    with gr.Column(scale=3, min_width=280):
                        history = gr.Radio(choices=history_files(), value=history_files()[0] if history_files() else None, label='Newest first', elem_id='history-list')
                    with gr.Column(scale=4, min_width=280):
                        history_audio = gr.Audio(value=load_history(history_files()[0]) if history_files() else None, show_label=False, interactive=False, buttons=['download'])
                        history_delete = gr.Button('Delete recording', size='sm', elem_classes='neon-delete', interactive=bool(history_files()))
                        history_status = gr.Markdown('')
            with gr.Column(visible=False) as settings_page:
                section('⚙', 'Settings')
                language = gr.Dropdown(choices=LANGUAGES, value='Auto', label='Language')
                speed = gr.Slider(0.5, 2, value=1, step=0.05, label='Speed (×)')
                pitch = gr.Slider(-12, 12, value=0, step=0.5, label='Pitch (semitones)')
                stability = gr.Slider(0, 1, value=0.75, step=0.05, label='Stability')
                tokens = gr.Slider(256, 4096, value=2048, step=256, label='Maximum generation tokens', info='Higher limits allow longer speech and increase generation time.')
                gr.Textbox(value=str(MODEL_PATH), label='Installed model', interactive=False)
                gr.Textbox(value=str(OUTPUT_DIR), label='Output folder', interactive=False)
                unload = gr.Button('Release GPU memory')
                settings_status = gr.Textbox(label='Status', interactive=False)
                folder = gr.Button('Open output folder')
                gr.Markdown('Filipino uses Auto (experimental). Stability adjusts sampling; speed and pitch are applied to the generated audio.')
                with gr.Accordion('About', open=False):
                    gr.Markdown('MB Cloning Voice · Qwen3-TTS 1.7B Base. Runs locally. Download the model separately and review the upstream model license.')
    footer_status = gr.Markdown('● Checking GPU…', elem_id='footer')
    pages = [clone_page, history_page, settings_page]
    navs = [nav_clone, nav_history, nav_settings]

    def navigate(index):
        return [gr.Column(visible=i == index) for i in range(3)] + [gr.Button(variant='primary' if i == index else 'secondary') for i in range(3)]

    for idx, nav in enumerate(navs):
        nav.click(lambda i=idx: navigate(i), outputs=pages + navs, queue=False, api_visibility='private')
    add_voice_upload.upload(import_uploaded_voice, inputs=add_voice_upload, outputs=[audio, voice_name, source_file, voice_status], queue=True, show_progress='hidden', api_visibility='private')
    audio.start_recording(lambda: '', outputs=source_file, queue=True)
    audio.clear(lambda: '', outputs=source_file, queue=True)
    for field in [voice_name, audio, consent, source_file]:
        field.change(save_readiness, inputs=[voice_name, audio, consent, source_file], outputs=[save, voice_status], queue=True, show_progress='hidden')
    library_version.change(save_readiness, inputs=[voice_name, audio, consent, source_file], outputs=[save, voice_status], queue=True, show_progress='hidden')
    for field in [script, active_voice, library_version]:
        field.change(generation_readiness, inputs=[script, active_voice], outputs=[button, generation_hint], queue=True, show_progress='hidden')
    current_output.change(lambda path: (gr.Button(interactive=bool(path)), gr.DownloadButton(value=path or None, visible=bool(path), interactive=bool(path))), inputs=current_output, outputs=[clear_audio, download_audio], queue=True, show_progress='hidden')
    history.change(lambda name: gr.Button(interactive=bool(name and name in history_files())), inputs=history, outputs=history_delete, queue=True, show_progress='hidden')
    save.click(save_voice, inputs=[voice_name, audio, transcript, consent, source_file], outputs=[library_version, voice_status, audio, voice_name, transcript, consent], api_visibility='private')
    script.change(lambda text: (f'{len(text or ""):,} / 10,000', gr.Button(interactive=bool(text))), inputs=script, outputs=[counter, clear_text], queue=True, show_progress='hidden')
    clear_text.click(lambda: '', outputs=script, queue=True, show_progress='hidden')
    button.click(generate_selected, inputs=[script, active_voice, language, speed, pitch, stability, tokens], outputs=[output, status, history, current_output], concurrency_limit=1, concurrency_id='generation', api_visibility='private')
    clear_audio.click(clear_result, outputs=[output, status, current_output], queue=True)
    history_delete.click(delete_recording, inputs=[history, current_output], outputs=[history, history_audio, history_status, output, status, current_output], queue=True, api_visibility='private')
    stop.click(request_stop, outputs=status, queue=False, api_visibility='private')
    folder.click(open_folder, outputs=settings_status, queue=False, api_visibility='private')
    nav_history.click(lambda: (history_update(), load_history(history_files()[0]) if history_files() else None), outputs=[history, history_audio], queue=True)
    history.change(load_history, inputs=history, outputs=history_audio, queue=True)
    unload.click(unload_model, outputs=settings_status, queue=False, api_visibility='private')
    demo.load(gpu_status, outputs=footer_status)
    gr.Timer(10).tick(gpu_status, outputs=footer_status, queue=False)

if __name__ == '__main__':
    from launch_mb_cloning_voice import available_port
    demo.queue(default_concurrency_limit=1)
    demo.launch(favicon_path=str(ROOT / 'MB_Cloning_Voice.ico'), server_name='localhost', server_port=available_port(), inbrowser=True, show_error=True,
                theme=gr.themes.Soft(primary_hue='purple', neutral_hue='slate'), css=CSS, allowed_paths=[str(VOICE_DIR)],
                head='<style>@media(min-width:1100px) and (min-height:750px){html,body,gradio-app{overflow:hidden!important;height:100dvh!important;min-height:0!important;}}</style>')
