# Chat Transcriber

I do a lot of coffee chats and networking calls, and I kept losing the details: the advice someone gave, the name of a team they mentioned, what I promised to follow up on. Notes during the call meant I wasn't really listening, so I started recording my calls instead and built this to turn the recordings into searchable, timestamped transcripts.

Point it at a recording (or a whole folder of them) and it writes a Markdown transcript next to each video. It runs fully offline on your own machine, so recordings never leave your computer.

## What it does

- Transcribes `.mkv`, `.mp4`, `.mov`, `.webm`, `.m4a`, `.mp3`, and `.wav` files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (an optimized OpenAI Whisper)
- Groups speech into timestamped paragraphs, breaking on pauses so speaker turns are easy to follow
- Skips recordings that already have a transcript, so you can rerun it on your recordings folder after every call
- Uses an NVIDIA GPU automatically when one is set up, and falls back to a multi-threaded CPU path otherwise
- Batches audio chunks and skips silence with voice activity detection to keep transcription fast

## Setup

Requires Python 3.9+.

```
pip install -r requirements.txt
```

Optional, for NVIDIA GPUs (much faster):

```
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

The first run downloads the speech model (~500 MB for the default `small.en`). After that it works offline.

## Usage

```
python transcribe_chats.py "C:\Users\me\Videos"          # every recording in a folder
python transcribe_chats.py "2026-11-17 18-10-18.mkv"     # one file
python transcribe_chats.py Videos --model base.en         # faster, slightly less accurate
python transcribe_chats.py Videos --accurate              # slower, best quality
```

| Option | Default | What it does |
|---|---|---|
| `--model` | `small.en` | `tiny.en`, `base.en`, `small.en`, `medium.en`, or `large-v3` |
| `--device` | `auto` | `auto` (GPU if it works, else CPU), `cuda`, or `cpu` |
| `--accurate` | off | Beam search decoding: slower, slightly more accurate |
| `--batch-size` | `8` | Audio chunks processed at once (`1` turns batching off) |
| `--force` | off | Redo recordings that already have a transcript |
| `--watch` | off | Keep running and transcribe each new recording once it's done saving |
| `--log` | none | Write output to a file (for running in the background with `pythonw`) |

## Run it automatically (Windows)

I keep the script in `Videos\chat-transcriber\` and run `setup_autostart.cmd` once. It registers a Task Scheduler task that starts the transcriber in watch mode every time I log in, so every recording OBS saves gets a transcript a couple of minutes after the call ends, with nothing to run by hand.

```
python transcribe_chats.py "C:\Users\me\Videos" --watch --log transcriber.log
```

Watch mode waits until a file hasn't changed for 2 minutes before touching it, so it never grabs a recording OBS is still writing.

## Output

`2026-11-17 18-10-18.mkv` → `2026-11-17 18-10-18.transcript.md`:

```markdown
# Transcript: 2026-11-17 18-10-18

- **Recorded:** Tue Nov 17, 2026 06:10 PM
- **Length:** 21:34
- **Source file:** 2026-11-17 18-10-18.mkv

---

**[00:03]** Hey, thanks for making time...

**[00:41]** ...
```

## A note on recording

Ask the other person before you record a call. It's good manners, and in some places it's the law.
